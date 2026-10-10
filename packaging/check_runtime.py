"""Container-side delivery checks. Does not modify algorithm files."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

assert torch.cuda.is_available(), "CUDA 不可用，请检查驱动和 NVIDIA Container Toolkit"
subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], check=True)
print(f"Python={sys.version.split()[0]} PyTorch={torch.__version__} CUDA={torch.version.cuda}", flush=True)
subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL)
subprocess.run(["ffprobe", "-version"], check=True, stdout=subprocess.DEVNULL)

root = Path("/app")
for relative in ("models/detection/detection_R39.pth", "models/reid/resnet50-0676ba61.pth",
                 "third_party/HyperLPR/resource/font/platech.ttf"):
    path = root / relative
    assert path.is_file() and path.stat().st_size, f"缺少资源: {path}"

# Exercise the compiled CUDA extension, rather than only importing it.
sys.path.insert(0, str(root / "algorithms/stmc"))
from src.tracker.solver import create_solver, multicut
edges = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.int64, device="cuda:0")
labels = multicut(edges, torch.tensor([1., 1., 1.], device="cuda:0"), create_solver("PD"))
torch.cuda.synchronize()
assert labels.numel() == 3 and torch.unique(labels).numel() == 1, "RAMA 运算结果异常"
print("RAMA CUDA 实际运算通过", flush=True)

os.environ["HYPERLPR_HOME"] = str(root / "models/license_plate/hyperlpr3")
sys.path.insert(0, str(root / "third_party/HyperLPR/Prj-Python"))
import hyperlpr3
catcher = hyperlpr3.LicensePlateCatcher(folder=os.environ["HYPERLPR_HOME"], detect_level=hyperlpr3.DETECT_LEVEL_LOW)
catcher(np.zeros((256, 512, 3), dtype=np.uint8))
print("车牌模型加载及空白图推理通过（不代表识别准确率验收）", flush=True)

if Path("/data/input").is_dir():
    videos = sorted(p for p in Path("/data/input").iterdir()
                    if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".m4v"})
    assert 1 <= len(videos) <= 4, f"需要1～4个视频，实际为{len(videos)}"
    assert len({p.stem for p in videos}) == len(videos), "视频相机名称重复"
    calibration_root = Path("/data/calibrations") if Path("/data/calibrations").is_dir() else root / "configs/cameras/r39"
    for video in videos:
        probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(video)], text=True)
        assert json.loads(probe)["streams"], f"没有视频流: {video.name}"
        candidates = [calibration_root / video.stem / "calibration.json", calibration_root / f"{video.stem}.json"]
        calibration = next((p for p in candidates if p.is_file()), None)
        assert calibration, f"未找到 {video.stem} 标定；设置 ROAD_CALIBRATION_DIR 或按说明显式指定相机 ID"
        matrix = np.asarray(json.loads(calibration.read_text())["homography"], dtype=float)
        assert matrix.shape == (3, 3) and np.isfinite(matrix).all() and np.linalg.matrix_rank(matrix) == 3, f"标定矩阵无效: {calibration}"
    with tempfile.TemporaryFile(dir="/data/output") as handle:
        handle.write(b"write-check")
        handle.flush()
    stat = os.statvfs("/data/output")
    print(f"输入/标定/写入检查通过；输出目录剩余 {stat.f_bavail * stat.f_frsize / 2**30:.1f} GiB")
