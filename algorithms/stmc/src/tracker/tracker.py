import json
import os
import statistics
import time
from typing import Any, List, Optional, Tuple

import motmetrics as mm
import torch
from omegaconf import DictConfig
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

from .similarities import batch_bev_distance, batch_cosine_similarity, batched_box_iou, cosine_similarity
from .solver import multicut, scale_weights
from .supertrack import SuperTrack, TrackState


class Tracker:
    def __init__(
        self,
        solver_opts: Any,
        cfg: DictConfig,
        n_cams: int,
        feature_extractor: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = "cpu",
    ):
        """
        Initialize the Tracker.

        Args:
            solver_opts: Options for the solver.
            cfg: Configuration dictionary.
            n_cams: Number of cameras.
            feature_extractor: Feature extractor module.
            device: Device to run the tracker on.
        """
        self.feature_extractor = feature_extractor
        self.solver_opts = solver_opts
        self.device = device

        self.current_data = None

        self.feature_dim = cfg.tracker.fdim
        self.n_cams = n_cams
        self.cfg = cfg.tracker

        self.tracks: List[SuperTrack] = []

        self.frame = 0
        self.free_id = 1

        self.latency = []

        self.update_interval = 1
        self.stats = {
            "# Killed": 0,
            "Latency": 0,
        }

        self.cumulative_execution_time = 0
        self.position_history = {}
        position_cfg = self.cfg.get("position_state") if hasattr(self.cfg, "get") else None
        self.active_cameras_only = bool(position_cfg.get("active_cameras_only", False)) if position_cfg is not None else False
        SuperTrack.use_active_positions = self.active_cameras_only
        self.short_recovery = self.cfg.get("short_recovery") if hasattr(self.cfg, "get") else None
        self.link_guard = self.cfg.get("link_guard") if hasattr(self.cfg, "get") else None
        self._guard_logs: list[str] = []
        self.knowledge = None
        knowledge_cfg = self.cfg.get("knowledge") if hasattr(self.cfg, "get") else None
        if knowledge_cfg is not None and bool(knowledge_cfg.get("enabled", False)):
            from .knowledge_prior import KnowledgePrior

            self.knowledge = KnowledgePrior(knowledge_cfg)
        self.motion_memory = None
        memory_cfg = self.cfg.get("motion_memory") if hasattr(self.cfg, "get") else None
        if memory_cfg is not None and bool(memory_cfg.get("enabled", False)):
            from .motion_memory import MotionMemory

            self.motion_memory = MotionMemory(memory_cfg)

    def step(self, sample):
        """
        Perform a single step of tracking.

        Args:
            sample: Input sample containing detections and features.

        Returns:
            tuple: A tuple containing current results and predicted results.
        """
        # move sample to device and remove batch dimension
        t0 = time.time()
        for key in sample.keys():
            if key != "images":
                sample[key] = sample[key].to(self.device).squeeze(0)
        self.frame += 1
        if self.frame % self.update_interval == 0:
            if sample["annotations"].size(0) > 0:
                matched, unmatched = self.update(sample)
                self._handle_unmatched(unmatched)

        t1 = time.time()
        self.cumulative_execution_time += t1 - t0
        self.latency.append(t1 - t0)

        self._sanitize()
        self._remember_real_observations()
        self._record_positions()

        rresults = self.get_result()

        self.predict()
        if self.knowledge is not None:
            self.knowledge.flush()
        if self.motion_memory is not None:
            self.motion_memory.flush(self.frame)
        if self.link_guard is not None and self.link_guard.get("log_path"):
            self._flush_guard(self.link_guard.get("log_path"))

        presults = self.get_result()

        return rresults, presults

    def update(self, sample):
        """
        Update the tracker with new detections and features.

        Args:
            sample: Input sample containing detections and features.

        Returns:
            tuple: A tuple containing matched and unmatched tracks.
        """
        features = self.feature_extractor(sample)
        raw_superboxes = self._new_superboxes_from_data(sample, features)
        high_boxes, low_conf_superboxes = self._split_confidence(raw_superboxes)
        superboxes = high_boxes

        relevant_tracks = self.tracks + superboxes
        _track_indices = torch.arange(len(self.tracks)).to(self.device)
        _superbox_indices = torch.arange(len(self.tracks), len(relevant_tracks)).to(self.device)

        # 低分框不进第一轮图 避免把高分边挤掉后再被丢掉
        low_conf_indices = None

        if len(relevant_tracks) == 0:
            return [], []

        timed = os.environ.get("STMC_TIME") == "1"
        clock = time.perf_counter if timed else None
        mark = clock() if timed else 0.0
        features, positions, boxes, where = self._association_inputs(relevant_tracks)
        if timed:
            spans = {"inputs": clock() - mark}
            mark = clock()

        # compute (n_tracks) x (n_tracks) similarity matrix
        similarities = self._compute_similarities(features, positions, boxes)
        if timed:
            spans["similar"] = clock() - mark
            mark = clock()
        if where is not None:
            appearance, position, iou_sim = similarities
            appearance, position = self._fill_unshared_cameras(appearance, position, features, positions, where)
            similarities = (appearance, position, iou_sim)
        if timed:
            spans["fill"] = clock() - mark
            mark = clock()

        # compute weighted graph
        rescale_thresh = self.cfg.matching.rescale_threshold
        dist_thresh = self.cfg.matching.distance_threshold
        iou_bias = self.cfg.prematching.iou_bias if self.cfg.prematching.enabled else 0
        edge_index, edge_weights = self._build_weighted_graph(
            relevant_tracks,
            similarities,
            rescale_thresh,
            dist_thresh,
            iou_bias,
            reid_decay=self.cfg.matching.reid_decay,
        )
        if timed:
            spans["graph"] = clock() - mark
            mark = clock()
        labels = multicut(edge_index, edge_weights, self.solver_opts)
        if timed:
            spans["cut"] = clock() - mark
            print(
                f"STMC_TIME frame {self.frame} n {len(relevant_tracks)} "
                + " ".join(f"{key} {value:.3f}" for key, value in spans.items()),
                flush=True,
            )
        if os.environ.get("STMC_DIAG") == "1" and self.frame in (102, 103, 116, 117, 118):
            self._write_diag(relevant_tracks, similarities, labels)

        matched_tracks, unmatched_tracks = self._match(relevant_tracks, labels, low_conf_indices=low_conf_indices)
        if low_conf_superboxes:
            matched_tracks, unmatched_tracks = self._attach_low(matched_tracks, unmatched_tracks, low_conf_superboxes)

        self.tracks = matched_tracks + unmatched_tracks
        return matched_tracks, unmatched_tracks

    def _attach_low(self, matched, unmatched, lows):
        """未匹配轨迹再用低分框补 1 到 3 帧 必须同类别 IoU 和外观都过线 且一对一"""
        existing = [track for track in unmatched if track.label is not None]
        held = [track for track in unmatched if track.label is None]
        if not existing:
            return matched, unmatched
        pairs = []
        for row, track in enumerate(existing):
            missed = int(self.frame - track.frame)
            if missed < 1 or missed > 3:
                continue
            for col, det in enumerate(lows):
                if int(track.class_id) != int(det.class_id):
                    continue
                overlap, reid = self._low_evidence(track, det)
                if overlap < 0.45 or reid < 0.80:
                    continue
                pairs.append((float(overlap + reid), row, col))
        pairs.sort(reverse=True)
        used_track = set()
        used_det = set()
        recovered = []
        for _score, row, col in pairs:
            if row in used_track or col in used_det:
                continue
            track = existing[row]
            track.update(lows[col])
            used_track.add(row)
            used_det.add(col)
            recovered.append(track)
            self._guard_log(track, lows[col], "low_recover", float(_score), float(_score))
        still = [track for index, track in enumerate(existing) if index not in used_track]
        return matched + recovered, still + held

    def _low_evidence(self, track: SuperTrack, det: SuperTrack) -> tuple[float, float]:
        best_iou = -1.0
        best_reid = -1.0
        for cam in range(self.n_cams):
            if torch.isnan(det.boxes[cam]).any() or torch.isnan(track.boxes[cam]).any():
                continue
            if torch.isnan(det.features[cam]).any() or torch.isnan(track.features[cam]).any():
                continue
            overlap = box_iou(track.tlbr[cam].view(1, 4), det.tlbr[cam].view(1, 4))
            left = track.features[cam]
            right = det.features[cam]
            reid = torch.dot(left, right) / (left.norm() * right.norm()).clamp(min=1e-6)
            if float(overlap) > best_iou:
                best_iou = float(overlap)
                best_reid = float(reid)
        return best_iou, best_reid

    def _handle_unmatched(self, unmatched_tracks):
        """
        Handle unmatched tracks by updating their inactive status.

        Args:
            unmatched_tracks: List of unmatched tracks.
        """
        for track in unmatched_tracks:
            for cam in range(self.n_cams):
                if track.track_where[cam]:
                    track.inactive_since[cam] += 1

    def predict(self):
        """
        Project existing tracks into the future.
        """
        for track in self.tracks:
            track.predict()

    def _new_superboxes_from_data(self, sample, sample_features):
        """
        Create new superboxes from detections and features.

        Args:
            sample: Input sample containing detection information.
            sample_features: Extracted features from the sample.

        Returns:
            list: List of new SuperTrack objects.
        """
        n_rows = sample_features.shape[0]

        features = torch.full((n_rows, self.n_cams, self.feature_dim), float("nan"), device=self.device)
        boxes = torch.full((n_rows, self.n_cams, 4), float("nan"), device=self.device)
        positions_2d = torch.full((n_rows, self.n_cams, 2), float("nan"), device=self.device)
        positions_3d = torch.full((n_rows, self.n_cams, 2), float("nan"), device=self.device)

        cam_ids = sample["annotations"][:, 0].int()
        features[torch.arange(n_rows), cam_ids] = sample_features
        boxes[torch.arange(n_rows), cam_ids] = sample["annotations"][:, 3:7]
        positions_2d[torch.arange(n_rows), cam_ids] = sample["positions_2d"]
        positions_3d[torch.arange(n_rows), cam_ids] = sample["positions_3d"]
        confidences = sample["annotations"][:, 7]
        class_ids = sample["annotations"][:, 10].int()

        superboxes = [
            SuperTrack(
                frame=self.frame,
                features=features[row],
                boxes=boxes[row],
                positions_2d=positions_2d[row],
                positions_3d=positions_3d[row],
                confidence=confidences[row],
                class_id=int(class_ids[row].item()),
            )
            for row in range(n_rows)
        ]

        return superboxes

    def _merge_tracks(self, tracks):
        """
        Merge multiple tracks into a single track.

        Args:
            tracks: List of tracks to merge.

        Returns:
            SuperTrack: Merged track.
        """
        _frames = sorted({track.frame for track in tracks})

        newest_frame = _frames[-1]
        if len(_frames) > 1:
            penult_frame = _frames[-2]

        assert tracks[-1].frame == newest_frame

        newest_evidence = [track for track in tracks if track.frame == newest_frame]

        features = (torch.ones(self.n_cams, self.feature_dim) * (torch.nan)).to(self.device)
        boxes = (torch.ones(self.n_cams, 4) * (torch.nan)).to(self.device)
        positions_2d = (torch.ones(self.n_cams, 2) * (torch.nan)).to(self.device)
        positions_3d = (torch.ones(self.n_cams, 2) * (torch.nan)).to(self.device)
        track_where = torch.zeros(self.n_cams, dtype=torch.bool).to(self.device)

        for cam_id in range(self.n_cams):
            for track in newest_evidence:
                if not torch.isnan(track.features[cam_id]).any():
                    features[cam_id] = track.features[cam_id]
                    boxes[cam_id] = track.boxes[cam_id]
                    positions_2d[cam_id] = track.positions_2d[cam_id]
                    positions_3d[cam_id] = track.positions_3d[cam_id]
                    track_where[cam_id] = True
                    break

        merged_track = SuperTrack(
            frame=newest_frame,
            features=features,
            boxes=boxes,
            positions_2d=positions_2d,
            positions_3d=positions_3d,
            class_id=tracks[0].class_id,
        )

        if len(_frames) > 1:
            penult_track = [track for track in tracks if track.frame == penult_frame][0]
            penult_track.update(merged_track)
            merged_track = penult_track

        return merged_track

    def _match(self, tracks, labels, low_conf_indices=None):
        """
        Match superboxes with superboxes, and merged superboxes with existing supertracks.

        Args:
            tracks: List of tracks to match.
            labels: Labels for each track.
            low_conf_indices: Indices of low confidence detections.

        Returns:
            tuple: A tuple containing new tracks and unmatched tracks.
        """
        new_tracks = []
        unmatched_tracks = []

        for label in torch.unique(labels):
            track_indices = torch.where(labels == label)[0].tolist()
            if len(track_indices) == 1:
                track = tracks[track_indices[0]]
                if low_conf_indices is not None and track_indices[0] in low_conf_indices:
                    continue
                if track.state == TrackState.CREATED:
                    new_tracks.append(track)
                else:
                    unmatched_tracks.append(track)
            else:
                if low_conf_indices is None:
                    relevant_tracks = sorted([tracks[i] for i in track_indices], key=lambda x: x.frame)
                else:
                    relevant_tracks = sorted(
                        [tracks[i] for i in track_indices if i not in low_conf_indices], key=lambda x: x.frame
                    )
                if not relevant_tracks:
                    continue
                merged_track = self._merge_tracks(relevant_tracks)
                if low_conf_indices is not None and not merged_track.is_complete():
                    relevant_low_conf_tracks = [tracks[i] for i in track_indices if i in low_conf_indices]
                    merged_track = self._merge_tracks([merged_track] + relevant_low_conf_tracks)
                new_tracks.append(merged_track)

        return new_tracks, unmatched_tracks

    @staticmethod
    def _compute_similarities(features, positions, boxes):
        """
        Compute similarity matrices for features, positions, and boxes.

        Args:
            features: Tensor of track features.
            positions: Tensor of track positions.
            boxes: Tensor of track bounding boxes.

        Returns:
            tuple: A tuple containing similarity matrices for features, positions, and IoU.
        """
        # permute to (n_cams, n_tracks, feature_dim), (n_cams, n_tracks, 2), (n_cams, n_tracks, 4)
        features = features.permute(1, 0, 2)
        positions = positions.permute(1, 0, 2)
        boxes = boxes.permute(1, 0, 2)

        # compute pairwise similarities (n_cams, n_tracks, n_tracks)
        feature_sim = batch_cosine_similarity(features, features)
        position_dist = batch_bev_distance(positions)
        iou_sim = batched_box_iou(boxes)

        # average-pool similarities to (n_tracks, n_tracks)
        feature_sim = torch.nanmean(feature_sim, dim=0)
        position_dist = torch.nanmean(position_dist, dim=0)
        iou_sim = torch.nanmean(iou_sim, dim=0)

        return feature_sim, position_dist, iou_sim

    def _build_weighted_graph(
        self,
        tracks: List[SuperTrack],
        similarities: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        rescale_thresh: float,
        dist_thresh: float,
        iou_bias: float,
        reid_decay: float = 1,
        penalty: float = -100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a weighted graph from tracks and similarity matrices.

        Args:
            tracks: List of tracks.
            similarities: Tuple of similarity matrices.
            rescale_thresh: Threshold for rescaling weights.
            dist_thresh: Distance threshold for feasibility.
            iou_bias: Bias to add for IoU-based matching.
            reid_decay: Decay factor for ReID scores.
            penalty: Penalty for infeasible edges.

        Returns:
            tuple: A tuple containing edge indices and edge weights of the graph.
        """
        adj = self._initialize_adjacency_matrix(similarities, tracks, reid_decay, rescale_thresh, dist_thresh)

        if self.cfg.prematching.enabled:
            adj = self._apply_prematching(adj, tracks, iou_bias)
        adj = self._apply_short_recovery(adj, tracks)
        adj = self._apply_link_guard(adj, tracks)
        if self.motion_memory is not None:
            adj = self.motion_memory.apply(adj, tracks, self.frame)
        if self.knowledge is not None:
            adj = self._apply_knowledge(adj, tracks)

        adj = self._finalize_adjacency_matrix(adj, penalty, tracks)

        return self._get_edge_index_and_weights(adj)

    def _initialize_adjacency_matrix(
        self,
        similarities: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tracks: List[SuperTrack],
        reid_decay: float,
        rescale_thresh: float,
        dist_thresh: float,
    ) -> torch.Tensor:
        """
        Initialize the adjacency matrix for the graph.

        Args:
            similarities: Tuple of similarity matrices.
            tracks: List of tracks.
            reid_decay: Decay factor for ReID scores.
            rescale_thresh: Threshold for rescaling weights.
            dist_thresh: Distance threshold for feasibility.

        Returns:
            torch.Tensor: Initialized adjacency matrix.
        """
        appearance_sim, position_dist, _ = similarities
        device = appearance_sim.device

        frame_support_pairs = [(track.frame, track.track_where) for track in tracks]
        frames, supports = zip(*frame_support_pairs)

        times = torch.tensor(frames, dtype=torch.int, device=device)
        lost = torch.tensor([track.state == TrackState.LOST for track in tracks], device=device)
        lost_since = torch.tensor([track.lost_since for track in tracks], device=device)

        appearance_sim = appearance_sim * reid_decay**lost_since
        appearance_sim = scale_weights(appearance_sim, rescale_thresh)

        combined_sim = self.cfg.matching.rescale_weight * appearance_sim + self.cfg.matching.distance_weight * (
            1 - position_dist / dist_thresh
        )

        adj = torch.zeros_like(appearance_sim)
        lmask = lost[:, None] | lost[None, :]
        same_time = times[:, None] == times[None, :]
        feasible = (position_dist < dist_thresh) | lmask

        adj[same_time & feasible] = torch.clip(combined_sim[same_time & feasible], min=0, max=1)
        adj[~same_time] = combined_sim[~same_time]
        adj[lmask] = combined_sim[lmask]

        return adj

    def _apply_prematching(self, adj: torch.Tensor, tracks: List[SuperTrack], iou_bias: float) -> torch.Tensor:
        """
        Apply prematching to the adjacency matrix.

        Args:
            adj: Adjacency matrix.
            tracks: List of tracks.
            iou_bias: Bias to add for IoU-based matching.

        Returns:
            torch.Tensor: Updated adjacency matrix after prematching.
        """
        cur_frame = max(track.frame for track in tracks)
        pen_frame = cur_frame - 1
        cur_track_idx_by_cam = [[] for _ in range(self.n_cams)]
        pen_track_idx_by_cam = [[] for _ in range(self.n_cams)]

        for i, track in enumerate(tracks):
            if track.frame == cur_frame:
                for cam in range(self.n_cams):
                    if not torch.isnan(track.boxes[cam]).any():
                        cur_track_idx_by_cam[cam].append(i)
            elif track.frame == pen_frame:
                for cam in range(self.n_cams):
                    if not torch.isnan(track.boxes[cam]).any():
                        pen_track_idx_by_cam[cam].append(i)

        for cam in range(self.n_cams):
            cur_boxes_cam = [tracks[i].tlbr[cam] for i in cur_track_idx_by_cam[cam]]
            pen_boxes_cam = [tracks[i].tlbr[cam] for i in pen_track_idx_by_cam[cam]]

            if not cur_boxes_cam or not pen_boxes_cam:
                continue

            iou_dist = 1 - box_iou(torch.stack(cur_boxes_cam), torch.stack(pen_boxes_cam))
            row_ind, col_ind = linear_sum_assignment(iou_dist.cpu().numpy())

            for r, c in zip(row_ind, col_ind):
                if iou_dist[r, c] > self.cfg.prematching.iou_threshold:
                    continue
                cur_idx = cur_track_idx_by_cam[cam][r]
                if self.cfg.prematching.prune_remaining:
                    adj[cur_idx] = 0
                    adj[:, cur_idx] = 0
                adj[cur_idx, pen_track_idx_by_cam[cam][c]] += iou_bias
                adj[pen_track_idx_by_cam[cam][c], cur_idx] += iou_bias

        return adj

    def _write_diag(self, tracks: List[SuperTrack], similarities, labels: torch.Tensor) -> None:
        """只在 STMC_DIAG=1 时写出指定帧的关联边 不改变权重"""
        appearance, position, _iou = similarities
        rescale_thresh = self.cfg.matching.rescale_threshold
        dist_thresh = self.cfg.matching.distance_threshold
        iou_bias = self.cfg.prematching.iou_bias if self.cfg.prematching.enabled else 0
        adj = self._initialize_adjacency_matrix(
            similarities, tracks, self.cfg.matching.reid_decay, rescale_thresh, dist_thresh
        )
        if self.cfg.prematching.enabled:
            adj = self._apply_prematching(adj, tracks, iou_bias)
        adj = self._apply_short_recovery(adj, tracks)
        adj = self._apply_link_guard(adj, tracks)
        if self.motion_memory is not None:
            adj = self.motion_memory.apply(adj, tracks, self.frame)
        base = adj.clone()
        adj = self._finalize_adjacency_matrix(adj, -100, tracks)
        path = os.environ.get("STMC_DIAG_PATH")
        if not path:
            return
        rows = []
        label_list = labels.detach().cpu().tolist()
        for left in range(len(tracks)):
            for right in range(left + 1, len(tracks)):
                older, newer = tracks[left], tracks[right]
                if older.frame == newer.frame:
                    continue
                if older.frame > newer.frame:
                    older, newer = newer, older
                    left_i, right_i = right, left
                else:
                    left_i, right_i = left, right
                if newer.frame != self.frame or newer.label is not None:
                    continue
                weight = float(adj[min(left, right), max(left, right)].item())
                if abs(weight) < 1e-6 and abs(float(base[left_i, right_i].item())) < 0.05:
                    continue
                slots = []
                for cam in range(self.n_cams):
                    pos = older.positions_3d[cam].detach().cpu()
                    slots.append(
                        {
                            "cam": cam,
                            "bev": None if torch.isnan(pos).any() else [float(pos[0]), float(pos[1])],
                            "where": bool(older.track_where[cam].item()),
                            "inactive": int(older.inactive_since[cam].item()),
                        }
                    )
                det = newer.positions_3d.detach().cpu()
                finite = ~torch.isnan(det).any(dim=-1)
                det_bev = None
                if bool(finite.any()):
                    point = det[finite][0]
                    det_bev = [float(point[0]), float(point[1])]
                rows.append(
                    {
                        "frame": int(self.frame),
                        "old_label": None if older.label is None else int(older.label),
                        "old_state": int(older.state),
                        "old_frame": int(older.frame),
                        "lost_since": int(older.lost_since),
                        "old_slots": slots,
                        "det_bev": det_bev,
                        "det_class": newer.class_id,
                        "old_class": older.class_id,
                        "appearance": float(appearance[left_i, right_i].item()),
                        "position_m": float(position[left_i, right_i].item()),
                        "weight_before_penalty": float(base[left_i, right_i].item()),
                        "weight": weight,
                        "same_group": int(label_list[left]) == int(label_list[right]),
                    }
                )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _association_inputs(self, tracks: List[SuperTrack]):
        """默认仍用 phantomize 打开开关后只让 track_where 为真的相机进入距离和外观"""
        boxes = torch.stack([track.tlbr for track in tracks])
        if not self.active_cameras_only:
            features = torch.stack([track.p_features for track in tracks])
            positions = torch.stack([track.p_positions for track in tracks])
            return features, positions, boxes, None
        features = torch.stack([track.features for track in tracks]).clone()
        positions = torch.stack([track.positions_3d for track in tracks]).clone()
        where = torch.stack([track.track_where for track in tracks])
        features[~where] = float("nan")
        positions[~where] = float("nan")
        return features, positions, boxes, where

    def _fill_unshared_cameras(self, appearance, position, features, positions, where):
        """没有共同有效相机时 用各自有效相机的均值做一次跨相机比较"""
        counts = where.sum(dim=1).clamp(min=1).unsqueeze(-1).to(features.dtype)
        feature_mean = torch.nan_to_num(features, nan=0.0).sum(dim=1) / counts
        position_mean = torch.nan_to_num(positions, nan=0.0).sum(dim=1) / counts
        fallback_appearance = cosine_similarity(feature_mean, feature_mean)
        fallback_position = torch.norm(position_mean[:, None, :] - position_mean[None, :, :], dim=-1)
        empty = where.sum(dim=1) == 0
        fallback_appearance[empty, :] = float("nan")
        fallback_appearance[:, empty] = float("nan")
        fallback_position[empty, :] = float("nan")
        fallback_position[:, empty] = float("nan")
        appearance = torch.where(torch.isnan(appearance), fallback_appearance, appearance)
        position = torch.where(torch.isnan(position), fallback_position, position)
        return appearance, position

    def _is_high_confidence(self, box: SuperTrack) -> bool:
        high_map = self.cfg.get("class_high_thresh") if hasattr(self.cfg, "get") else None
        if high_map is not None and box.class_id is not None:
            return float(box.confidence) >= float(high_map[int(box.class_id)])
        return float(box.confidence) >= float(self.cfg.confidence_thresh)

    def _split_confidence(self, boxes: List[SuperTrack]) -> tuple[list, list]:
        """高分框可以新建轨迹 低分框只参与关联"""
        if self.cfg.low_confidence_thresh is None:
            return [box for box in boxes if self._is_high_confidence(box)], []
        floor = float(self.cfg.low_confidence_thresh)
        high, low = [], []
        for box in boxes:
            if self._is_high_confidence(box):
                high.append(box)
            elif float(box.confidence) >= floor:
                low.append(box)
        return high, low

    def _guard_overlaps(self, tracks: List[SuperTrack], older: list[int], current: list[int]):
        """每个旧轨迹和当前检测在共同相机上的最大预测 IoU"""
        import numpy as np

        overlaps = np.full((len(older), len(current)), np.nan, dtype=np.float64)
        for cam in range(self.n_cams):
            current_pos = []
            current_box = []
            for col, index in enumerate(current):
                if bool(torch.isnan(tracks[index].boxes[cam]).any()):
                    continue
                current_pos.append(col)
                current_box.append(tracks[index].tlbr[cam])
            older_pos = []
            older_box = []
            for row, index in enumerate(older):
                track = tracks[index]
                if int(track.last_real_frame[cam]) < 0:
                    continue
                gap = int(self.frame - int(track.last_real_frame[cam]))
                if gap <= 0:
                    continue
                predicted = track.last_real_boxes[cam] + track.velocities_2d[cam] * gap
                if bool((predicted[2] <= 0) or (predicted[3] <= 0)):
                    predicted = track.last_real_boxes[cam]
                boxes = [predicted]
                if gap > 1:
                    boxes.append(track.last_real_boxes[cam])
                for source in boxes:
                    predicted_tlbr = source.clone()
                    predicted_tlbr[2] = source[0] + source[2]
                    predicted_tlbr[3] = source[1] + source[3]
                    older_pos.append(row)
                    older_box.append(predicted_tlbr)
            if not current_box or not older_box:
                continue
            values = box_iou(torch.stack(older_box), torch.stack(current_box)).detach().cpu().numpy()
            for row_i, row in enumerate(older_pos):
                for col_i, col in enumerate(current_pos):
                    value = float(values[row_i, col_i])
                    previous = overlaps[row, col]
                    overlaps[row, col] = value if previous != previous else max(previous, value)
        return overlaps

    def _apply_link_guard(self, adj: torch.Tensor, tracks: List[SuperTrack]) -> torch.Tensor:
        """近距离候选分不清时宁可断开 也不把两个目标连成同一条"""
        cfg = self.link_guard
        if cfg is None or not bool(cfg.get("enabled", False)):
            return adj
        import numpy as np

        from .link_guard import competition_clear

        max_gap = int(cfg.get("max_gap", 3))
        margin = float(cfg.get("ambiguity_margin", 0.12))
        iou_decide = float(cfg.get("iou_decide", 0.15))
        reject_iou = float(cfg.get("reject_iou", 0.05))
        current = [index for index, track in enumerate(tracks) if track.frame == self.frame and track.label is None]
        older = []
        for index, track in enumerate(tracks):
            if track.label is None:
                continue
            missed = int(self.frame - track.frame) - 1
            if 0 <= missed <= max_gap:
                older.append(index)
        if not current or not older:
            return adj
        base = adj.detach().cpu().numpy()
        matrix = base[np.ix_(older, current)].astype(np.float64)
        overlaps = self._guard_overlaps(tracks, older, current)
        for row, left in enumerate(older):
            for col, right in enumerate(current):
                if int(tracks[left].class_id) != int(tracks[right].class_id):
                    matrix[row, col] = 0.0
                    continue
                overlap = overlaps[row, col]
                rival = np.nanmax(overlaps[:, col]) if overlaps.shape[0] else overlap
                beaten = np.isfinite(rival) and np.isfinite(overlap) and float(rival - overlap) >= iou_decide
                if np.isfinite(overlap) and overlap < reject_iou and matrix[row, col] > 0 and beaten:
                    matrix[row, col] = 0.0
                    adj[left, right] = 0
                    adj[right, left] = 0
                    self._guard_log(tracks[left], tracks[right], "reject_motion", float(base[left, right]), overlap)
        filled = np.where(np.isfinite(overlaps), overlaps, -1.0)
        compete = matrix.copy()
        compete[~np.isfinite(overlaps)] = 0.0
        clear = competition_clear(compete, filled, margin, iou_decide)
        for row, left in enumerate(older):
            for col, right in enumerate(current):
                if not clear[row, col] or float(adj[left, right]) <= 0:
                    continue
                self._guard_log(tracks[left], tracks[right], "ambiguous", float(adj[left, right]), float(filled[row, col]))
                adj[left, right] = 0
                adj[right, left] = 0
        return adj

    def _guard_log(self, older: SuperTrack, newer: SuperTrack, action: str, weight: float, overlap: float) -> None:
        cfg = self.link_guard
        if cfg is None:
            return
        path = cfg.get("log_path")
        if not path:
            return
        record = {
            "frame": int(self.frame),
            "action": action,
            "older_id": None if older.label is None else int(older.label),
            "class_id": None if newer.class_id is None else int(newer.class_id),
            "weight": round(float(weight), 4),
            "pred_iou": None if overlap != overlap else round(float(overlap), 4),
        }
        self._guard_logs.append(json.dumps(record, ensure_ascii=False) + "\n")
        if len(self._guard_logs) >= 40:
            self._flush_guard(path)

    def _flush_guard(self, path) -> None:
        if not self._guard_logs:
            return
        from pathlib import Path

        target = Path(str(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as file:
            file.write("".join(self._guard_logs))
        self._guard_logs.clear()

    def _apply_short_recovery(self, adj: torch.Tensor, tracks: List[SuperTrack]) -> torch.Tensor:
        """漏检 1 到 3 帧时 用最后真实框的预测 IoU 加分 图像对不上的近邻不加正边"""
        cfg = self.short_recovery
        if cfg is None or not bool(cfg.get("enabled", False)):
            return adj
        max_gap = int(cfg.get("max_gap", 3))
        bonus = float(cfg.get("iou_bonus", 0.60))
        match_iou = float(cfg.get("match_iou", 0.50))
        reject_iou = float(cfg.get("reject_iou", 0.30))
        current_idx = [index for index, track in enumerate(tracks) if track.frame == self.frame and track.label is None]
        older_idx = []
        for index, track in enumerate(tracks):
            if track.label is None:
                continue
            missed = int(self.frame - track.frame) - 1
            if 1 <= missed <= max_gap:
                older_idx.append(index)
        if not current_idx or not older_idx:
            return adj
        best = torch.full((len(current_idx), len(older_idx)), -1.0, device=adj.device)
        last_real_frame = torch.stack([track.last_real_frame for track in tracks]).detach().cpu()
        box_missing = torch.isnan(torch.stack([track.boxes for track in tracks])).any(dim=-1).detach().cpu()
        for cam in range(self.n_cams):
            current_rows = [row for row, index in enumerate(current_idx) if not bool(box_missing[index, cam])]
            older_cols = [col for col, index in enumerate(older_idx) if int(last_real_frame[index, cam]) >= 0]
            if not current_rows or not older_cols:
                continue
            current_boxes = torch.stack([tracks[current_idx[row]].tlbr[cam] for row in current_rows])
            older_boxes = []
            for col in older_cols:
                older = tracks[older_idx[col]]
                gap = int(self.frame - older.frame)
                predicted = older.last_real_boxes[cam] + older.velocities_2d[cam] * gap
                invalid = ((predicted[2] <= 0) | (predicted[3] <= 0)).view(1)
                predicted = torch.where(invalid, older.last_real_boxes[cam].view(1, 4), predicted.view(1, 4)).view(4)
                predicted_tlbr = predicted.clone()
                predicted_tlbr[2] = predicted[0] + predicted[2]
                predicted_tlbr[3] = predicted[1] + predicted[3]
                older_boxes.append(predicted_tlbr)
            overlap = box_iou(current_boxes, torch.stack(older_boxes))
            for row_i, row in enumerate(current_rows):
                best[row, older_cols] = torch.maximum(best[row, older_cols], overlap[row_i])
        best_cpu = best.detach().cpu()
        weight_cpu = adj.detach().cpu()
        one_to_one = bool(cfg.get("one_to_one", False))
        bonus_ok = None
        if one_to_one:
            import numpy as np

            from .link_guard import mutual_keep

            gated = best_cpu.numpy().copy()
            gated[gated < match_iou] = -1.0
            bonus_ok = mutual_keep(gated, float(cfg.get("one_to_one_margin", 0.08)))
        for row, left in enumerate(current_idx):
            for col, right in enumerate(older_idx):
                value = float(best_cpu[row, col])
                if value < 0:
                    continue
                base_weight = float(weight_cpu[left, right])
                if base_weight != base_weight:
                    base_weight = 0.0
                if value >= match_iou and (bonus_ok is None or bool(bonus_ok[row, col])):
                    updated = base_weight + bonus
                elif value < reject_iou and base_weight > 0:
                    updated = 0.0
                else:
                    continue
                adj[left, right] = updated
                adj[right, left] = updated
        return adj

    def _remember_real_observations(self) -> None:
        """关联完成后记下本帧真实位置 预测发生在这之后 不会记进去"""
        if self.motion_memory is None:
            return
        for track in self.tracks:
            if track.label is None or int(track.frame) != int(self.frame):
                continue
            positions = track.positions_3d
            if positions.shape[-1] > 2:
                positions = positions[..., :2]
            masked = positions.clone()
            masked[~track.track_where] = float("nan")
            finite = ~torch.isnan(masked).any(dim=-1)
            if not bool(finite.any()):
                continue
            point = masked[finite].mean(dim=0).detach().cpu().numpy()
            self.motion_memory.remember(int(track.label), int(self.frame), point)

    def _mean_bev(self, track: SuperTrack):
        positions = track.positions_3d
        finite = ~torch.isnan(positions).any(dim=-1)
        if not bool(finite.any()):
            return None
        return positions[finite].mean(dim=0).detach().cpu().numpy()

    def _record_positions(self):
        """记录已经完成关联的位置 供下一帧检索 不写入当前帧还没判定的检测"""
        current = []
        for track in self.tracks:
            if track.label is None or track.state == TrackState.KILLED:
                continue
            point = self._mean_bev(track)
            if point is None:
                continue
            history = self.position_history.setdefault(int(track.label), [])
            frame = int(track.frame)
            if history and history[-1][0] == frame:
                history[-1] = (frame, point)
            else:
                history.append((frame, point))
            if len(history) > 40:
                del history[:-40]
            current.append(int(track.label))
        stale = [label for label in self.position_history if label not in current and label not in {
            int(track.label) for track in self.tracks if track.label is not None and track.state == TrackState.LOST
        }]
        for label in stale:
            self.position_history.pop(label, None)

    def _apply_knowledge(self, adj: torch.Tensor, tracks: List[SuperTrack]) -> torch.Tensor:
        """只改跨帧边 硬互斥留在后面的惩罚里"""
        for left in range(len(tracks)):
            for right in range(left + 1, len(tracks)):
                older = tracks[left]
                newer = tracks[right]
                if older.frame == newer.frame:
                    continue
                if older.frame > newer.frame:
                    older, newer = newer, older
                if newer.frame != self.frame or newer.label is not None or older.label is None:
                    continue
                history = [
                    point
                    for frame, point in self.position_history.get(int(older.label), [])
                    if frame < self.frame
                ]
                candidate = self._mean_bev(newer)
                if candidate is None or len(history) == 0:
                    continue
                base = float(adj[left, right].item())
                delta, _decision = self.knowledge.adjust(
                    history,
                    candidate,
                    base,
                    int(newer.frame - older.frame),
                )
                if delta == 0.0:
                    continue
                adj[left, right] = adj[left, right] + delta
                adj[right, left] = adj[right, left] + delta
        return adj

    def _finalize_adjacency_matrix(self, adj: torch.Tensor, penalty: float, tracks: List[SuperTrack]) -> torch.Tensor:
        """
        Finalize the adjacency matrix by applying penalties.

        Args:
            adj: Adjacency matrix.
            penalty: Penalty value for infeasible edges.
            tracks: List of tracks.

        Returns:
            torch.Tensor: Finalized adjacency matrix.
        """
        frame_support_pairs = [(track.frame, track.track_where) for track in tracks]
        frames, supports = zip(*frame_support_pairs)

        times = torch.tensor(frames, dtype=torch.int, device=adj.device)
        supps = torch.stack(supports).to(adj.device)

        same_time = times[:, None] == times[None, :]
        same_supp = (supps[:, None] & supps[None, :]).any(dim=2)
        class_ids = torch.tensor(
            [track.class_id for track in tracks], dtype=torch.int, device=adj.device
        )
        class_mismatch = class_ids[:, None] != class_ids[None, :]

        adj[same_time & same_supp] = penalty
        adj[class_mismatch] = penalty
        if self.active_cameras_only:
            adj = torch.nan_to_num(adj, nan=0.0)
        adj = adj * torch.triu(torch.ones_like(adj), diagonal=1)

        return adj

    def _get_edge_index_and_weights(self, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract edge indices and weights from the adjacency matrix.

        Args:
            adj: Adjacency matrix.

        Returns:
            tuple: A tuple containing edge indices and edge weights.
        """
        edge_index = torch.nonzero(adj).t().long()
        edge_weights = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_weights

    def _sanitize(self):
        """
        Sanitize the tracker by updating track states and removing killed tracks.
        """
        keep = []
        for k, track in enumerate(self.tracks):
            if track.state is TrackState.CREATED:
                track.activate()
            if track.label is None:
                track.set_label(self.free_id)
                self.free_id += 1
            if torch.all(~track.track_where):
                if torch.all(track.inactive_since[track.inactive_since > 0] > self.cfg.patience):
                    track.deactivate()
            if track.state is TrackState.LOST:
                if track.lost_since > self.cfg.memory:
                    track.kill()
                else:
                    track.lost_since += 1
            if track.state is not TrackState.KILLED:
                keep.append(track)
            for cam in range(self.n_cams):
                if track.inactive_since[cam] > self.cfg.patience:
                    track.reset([cam])
        killed = len(self.tracks) - len(keep)
        self.tracks = keep
        self.stats["# Tracks"] = len(self.tracks)
        self.stats["# Lost"] = len([track for track in self.tracks if track.state == TrackState.LOST])
        self.stats["# Killed"] += killed

        latency = statistics.mean(self.latency) if len(self.latency) > 0 else 0
        self.stats["FPS"] = int(1 / latency) if latency > 0 else 0

    def _get_active_tracks(self):
        """
        Get a list of active tracks.

        Returns:
            list: List of active tracks.
        """
        return [track for track in self.tracks if track.state != TrackState.KILLED]

    def get_result(self, normalization=None, scale=1.0):
        """
        Get the current online state of the tracker.

        Args:
            normalization: Optional normalization parameters.
            scale: Scale factor for the results.

        Returns:
            torch.Tensor: Tensor containing the current tracker state.
        """
        to_stack = [track.to_tensor() for track in self.tracks if track.state == TrackState.ACTIVE]
        if len(to_stack) > 0:
            result = torch.cat(to_stack)
        else:
            result = torch.empty(0)
        if result.size(0) > 0:
            if normalization is not None:
                min_x, min_y, max_x, max_y = normalization
                result[:, 7:9] = result[:, 7:9] * torch.tensor([max_x - min_x, max_y - min_y]) + torch.tensor(
                    [min_x, min_y]
                )
            result[:, 7:9] *= scale
        return result

    def _get_index_by_id(self, tid):
        """
        Get the index of a track by its ID.

        Args:
            tid: Track ID to search for.

        Returns:
            int: Index of the track with the given ID, or None if not found.
        """
        for i, track in enumerate(self.tracks):
            if track.label == tid:
                return i
        return None


def create_tracker(cfg, solver_cfg, feature_extractor, n_cams, device, writer=None):
    """
    Create a new Tracker instance.

    Args:
        cfg: Configuration dictionary.
        solver_cfg: Solver configuration.
        feature_extractor: Feature extractor module.
        n_cams: Number of cameras.
        device: Device to run the tracker on.
        writer: Optional writer for logging.

    Returns:
        Tracker: A new Tracker instance.
    """
    return Tracker(
        solver_opts=solver_cfg,
        cfg=cfg,
        feature_extractor=feature_extractor,
        n_cams=n_cams,
        device=device,
    )
