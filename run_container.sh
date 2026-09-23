#!/usr/bin/env bash
set -euo pipefail

IMAGE="roadside-perception:20260923"
if (( $# < 2 )); then
    echo "用法: $0 <视频目录> <输出目录> [流水线参数]" >&2
    echo "例如: $0 /data/videos /data/result --max-frames 200 --overwrite" >&2
    exit 2
fi

input_dir=$(realpath -e "$1")
if [[ ! -d "$input_dir" ]]; then
    echo "视频目录不存在: $input_dir" >&2
    exit 2
fi
mkdir -p "$2"
output_dir=$(realpath -e "$2")
shift 2

docker_args=(
    run --rm --gpus all --shm-size=8g
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp
    -v "$input_dir:/data/input:ro"
    -v "$output_dir:/data/output"
)
pipeline_args=(--input-dir /data/input --output /data/output)
if [[ -n "${ROAD_CALIBRATION_DIR:-}" ]]; then
    calibration_dir=$(realpath -e "$ROAD_CALIBRATION_DIR")
    docker_args+=(-v "$calibration_dir:/data/calibrations:ro")
    pipeline_args+=(--calibration-dir /data/calibrations)
fi

exec docker "${docker_args[@]}" "$IMAGE" "${pipeline_args[@]}" "$@"
