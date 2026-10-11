import json
import os
from subprocess import PIPE, run

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from src.datasets.dataset import create_dataloader
from src.tracker.encoder import create_encoder
from src.tracker.solver import create_solver
from src.tracker.tracker import create_tracker
from src.utils.evaluate import evaluate_tracker
from src.utils.export_json import TrackingJsonExporter, load_class_lookup
from src.utils.iotools import ResultsWriter
from src.utils.visualize import save_tracking_visualization


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.device == "cpu" or not torch.cuda.is_available():
        raise ValueError("This code runs on CUDA only. Please set device to 'cuda'.")
    else:
        device = torch.device(cfg.device)
        logger.info(f"🚀 Using device: {device}")

    cfg.tracker.matching.distance_weight = 1 - cfg.tracker.matching.rescale_weight

    # create output directories
    output_path = os.path.join(cfg.output_path)
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, cfg.dataset.name)
    logger.info(f"📂 Writing to output path: {output_path}")

    # Initialize wandb and tensorboard
    if cfg.logging.wandb.enable:
        wandb.init(project=cfg.logging.wandb.project)
        wandb.config.update(OmegaConf.to_container(cfg))
        if cfg.logging.wandb.tags is not None:
            wandb.run.tags = cfg.logging.wandb.tags

    # Initialize solver
    solver_opts = create_solver(cfg.solver.backend)
    logger.info(f"✨ Initialized solver, using backend: {cfg.solver.backend}")

    # Initialize dataset and dataloader
    dataloader = create_dataloader(cfg)
    logger.info("✨ Created dataloader.")

    # Initialize encoder
    encoder = create_encoder(cfg.encoder, device)
    logger.info("✨ Created encoder.")

    tracker = create_tracker(cfg, solver_opts, encoder, len(dataloader.dataset.camera_names), device)
    logger.info("✨ Initialized tracker.")

    results_writer = ResultsWriter(
        output_path=output_path,
        cfg=cfg,
        normalization=dataloader.dataset._norm_factors,
        camera_names=dataloader.dataset.camera_names,
    )

    json_exporter = None
    class_lookup = None
    class_match_iou = 0.3
    if cfg.get("export_json") is not None:
        class_match_iou = float(cfg.export_json.get("class_match_iou", 0.3))
        if cfg.export_json.enable:
            json_exporter = TrackingJsonExporter(
                output_path=output_path,
                camera_names=dataloader.dataset.camera_names,
                dataset_name=cfg.dataset.name,
                scene_path=cfg.dataset.scene_path,
                class_metadata_path=cfg.export_json.class_metadata_path,
                class_match_iou=class_match_iou,
            )
            class_lookup = json_exporter.class_lookup
            logger.info("✨ JSON exporter enabled.")
        elif cfg.export_json.class_metadata_path and cfg.visuals.plot_results:
            class_lookup = load_class_lookup(cfg.export_json.class_metadata_path)

    tw = tqdm(total=len(dataloader), desc="跟踪")
    for i, batch in enumerate(dataloader):
        results, _ = tracker.step(batch)
        results_writer.add(results)
        if json_exporter is not None:
            json_exporter.add(results)
        if (
            cfg.visuals.plot_results
            and cfg.visuals.store_files
            and (i + 1) % cfg.visuals.plot_interval == 0
        ):
            save_tracking_visualization(
                images=batch["images"],
                results=results,
                camera_names=dataloader.dataset.camera_names,
                output_dir=os.path.join(output_path, "visualizations"),
                frame_id=i + 1,
                class_lookup=class_lookup,
                class_match_iou=class_match_iou,
            )
        stats = tracker.stats
        tw.set_postfix(stats, refresh=False)
        tw.update()
        diag_max = os.environ.get("STMC_DIAG_MAX")
        if diag_max and tracker.frame >= int(diag_max):
            break

        if cfg.logging.wandb.enable:
            _stats_str_to_float = {k: float(v) for k, v in stats.items()}
            wandb.log(_stats_str_to_float, step=i)

    tw.close()

    logger.info(f"🕒 Cumulative execution time of tracker {tracker.cumulative_execution_time * 10}")
    logger.info(f"🕒 Average time per frame {tracker.cumulative_execution_time / tracker.frame}")

    results_writer.save()
    if json_exporter is not None:
        json_path = json_exporter.save()
        logger.info(f"📄 Structured tracking JSON: {json_path}")

    logger.info("🚀 Tracking completed.")
    logger.info(
        f"📈 Results saved to {results_writer.results_file}. "
        "Use the official evaluation script of the dataset for evaluation."
    )


if __name__ == "__main__":
    main()
