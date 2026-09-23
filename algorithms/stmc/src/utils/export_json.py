"""Export STMC multi-camera tracking results to structured JSON."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger


def _tlwh_to_xyxy(x: float, y: float, w: float, h: float) -> Dict[str, float]:
    return {"x0": x, "y0": y, "x1": x + w, "y1": y + h}


def _iou_xyxy(a: Dict[str, float], b: List[float]) -> float:
    """IoU between xyxy dict and xyxy list [x0,y0,x1,y1]."""
    ax0, ay0, ax1, ay1 = a["x0"], a["y0"], a["x1"], a["y1"]
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / (area_a + area_b - inter + 1e-6)


def _parse_camera_from_file_name(file_name: str) -> Optional[str]:
    # e.g. "R39_Aw_CamS/sam3_multi_frames/00000.jpg"
    parts = Path(file_name).parts
    return parts[0] if parts else None


def load_class_lookup(metadata_path: str) -> Dict[Tuple[str, int], List[dict]]:
    """
    Build (camera_key, frame_1based) -> [{bbox_xyxy, class_name, class_id, score}, ...].

    feature_metadata uses image_index / 00000.jpg (0-based);
    STMC frames are 1-based (000001.jpg).
    """
    path = Path(metadata_path)
    if not path.exists():
        logger.warning(f"Class metadata not found: {metadata_path}")
        return {}

    with open(path, "r") as f:
        meta = json.load(f)

    records = meta.get("records", meta if isinstance(meta, list) else [])
    lookup: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for rec in records:
        cam = _parse_camera_from_file_name(rec.get("file_name", ""))
        if cam is None:
            continue
        # 优先从文件名解析每相机帧号（00000.jpg -> frame 1）。
        # 注意：部分 feature_metadata 的 image_index 是跨相机全局序号
        #（Aw:0-1199, Bn:1200-2399, ...），不能直接当帧号用。
        stem = Path(rec.get("file_name", "")).stem
        if stem.isdigit():
            frame_1based = int(stem) + 1
        elif "image_index" in rec:
            frame_1based = int(rec["image_index"]) + 1
        else:
            continue

        lookup[(cam, frame_1based)].append(
            {
                "bbox_xyxy": rec.get("bbox_xyxy"),
                "class_name": rec.get("class_name"),
                "class_id": rec.get("class_id"),
                "score": rec.get("score"),
            }
        )

    logger.info(f"📚 Loaded class metadata from {metadata_path} ({len(records)} records).")
    return lookup


def lookup_class(
    class_lookup: Dict[Tuple[str, int], List[dict]],
    camera_key: str,
    frame: int,
    bbox_xyxy: Dict[str, float],
    iou_thresh: float = 0.5,
) -> Tuple[Optional[str], Optional[int], Optional[float]]:
    cands = class_lookup.get((camera_key, frame), [])
    best_iou, best = 0.0, None
    for c in cands:
        box = c.get("bbox_xyxy")
        if not box or len(box) != 4:
            continue
        iou = _iou_xyxy(bbox_xyxy, box)
        if iou > best_iou:
            best_iou, best = iou, c
    if best is None or best_iou < iou_thresh:
        return None, None, None
    return best.get("class_name"), best.get("class_id"), best.get("score")


class TrackingJsonExporter:
    """Accumulate per-frame STMC results and write a structured JSON."""

    def __init__(
        self,
        output_path: str,
        camera_names: List[str],
        dataset_name: str,
        scene_path: str,
        class_metadata_path: Optional[str] = None,
        class_match_iou: float = 0.5,
    ):
        self.output_file = os.path.join(output_path, "tracking_results.json")
        self.camera_names = list(camera_names)
        self.dataset_name = dataset_name
        self.scene_path = scene_path
        self.class_match_iou = class_match_iou
        self.class_lookup = (
            load_class_lookup(class_metadata_path) if class_metadata_path else {}
        )
        self.records: List[dict] = []

    def add(self, results: torch.Tensor) -> None:
        """
        Add one frame of tracker outputs.

        results rows: [cam_idx, track_id, frame, x, y, w, h, bev_x, bev_y]
        """
        if results is None or results.numel() == 0:
            return
        results = results.detach().cpu()
        for row in results:
            cam_idx = int(row[0].item())
            track_id = int(row[1].item())
            frame = int(row[2].item())
            x, y, w, h = [float(v) for v in row[3:7].tolist()]
            bev_x, bev_y = float(row[7].item()), float(row[8].item())
            camera_key = self.camera_names[cam_idx]
            bbox_xyxy = _tlwh_to_xyxy(x, y, w, h)

            class_name, class_id, score = None, None, None
            if self.class_lookup:
                class_name, class_id, score = lookup_class(
                    self.class_lookup,
                    camera_key,
                    frame,
                    bbox_xyxy,
                    self.class_match_iou,
                )

            self.records.append(
                {
                    "camera_key": camera_key,
                    "track_id": track_id,
                    "frame": frame,  # STMC 1-based
                    "frame_index": frame - 1,  # 与 box_info 的 0-based 对齐
                    "class_name": class_name,
                    "class_id": class_id,
                    "score": score,
                    "bbox_tlwh": [x, y, w, h],
                    "bbox_xyxy": bbox_xyxy,
                    "bev": {"x": bev_x, "y": bev_y},
                }
            )

    def save(self) -> str:
        payload = {
            "dataset": self.dataset_name,
            "scene_path": self.scene_path,
            "cameras": self.camera_names,
            "num_records": len(self.records),
            "frame_index_base": 0,
            "frame_base": 1,
            "note": (
                "frame 为 STMC 1-based（对应 000001.jpg）；"
                "frame_index = frame - 1，可与 box_info.aligned_pixel_bbox 对齐。"
            ),
            "records": self.records,
        }
        os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)
        with open(self.output_file, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
        logger.info(f"💾 Wrote tracking JSON: {self.output_file} ({len(self.records)} records)")
        return self.output_file


def associate_with_box_info(
    tracking_json_path: str,
    box_info_path: str,
    output_path: str,
    iou_thresh: float = 0.3,
) -> str:
    """
    Associate STMC tracking JSON with box_info/aligned_pixel_bbox.json.

    Match key: (camera_key, frame_index) + max IoU between bboxes.
    Writes a merged JSON with both stmc track_id and box_info global/local ids.
    """
    with open(tracking_json_path) as f:
        track_data = json.load(f)
    with open(box_info_path) as f:
        box_info = json.load(f)
    if isinstance(box_info, dict) and "pixel_boxes" in box_info:
        box_info = box_info["pixel_boxes"]

    # index box_info
    bi_index: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for rec in box_info:
        bi_index[(rec["camera_key"], int(rec["frame_index"]))].append(rec)

    merged = []
    matched = 0
    for rec in track_data["records"]:
        key = (rec["camera_key"], int(rec["frame_index"]))
        cands = bi_index.get(key, [])
        best_iou, best = 0.0, None
        for c in cands:
            iou = _iou_xyxy(rec["bbox_xyxy"], [c["bbox_xyxy"][k] for k in ("x0", "y0", "x1", "y1")])
            if iou > best_iou:
                best_iou, best = iou, c

        out = dict(rec)
        if best is not None and best_iou >= iou_thresh:
            matched += 1
            out["match_iou"] = best_iou
            out["box_info_global_obj_id"] = best.get("global_obj_id")
            out["box_info_local_obj_id"] = best.get("local_obj_id")
            out["box_info_class_name"] = best.get("class_name")
            out["box_info_label"] = best.get("label")
            if out.get("class_name") is None:
                out["class_name"] = best.get("class_name")
        else:
            out["match_iou"] = None
            out["box_info_global_obj_id"] = None
            out["box_info_local_obj_id"] = None
            out["box_info_class_name"] = None
            out["box_info_label"] = None
        merged.append(out)

    payload = {
        "tracking_json": tracking_json_path,
        "box_info": box_info_path,
        "iou_thresh": iou_thresh,
        "num_tracking_records": len(track_data["records"]),
        "num_matched": matched,
        "match_rate": matched / max(len(track_data["records"]), 1),
        "records": merged,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    logger.info(
        f"🔗 Association saved to {output_path} "
        f"(matched {matched}/{len(track_data['records'])} = {payload['match_rate']:.2%})"
    )
    return output_path
