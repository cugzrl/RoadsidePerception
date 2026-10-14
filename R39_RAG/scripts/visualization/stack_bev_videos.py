"""把同一时间轴的BEV视频横向拼在一起"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--column-width", type=int, default=1280)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.inputs) != len(args.labels):
        raise ValueError("视频和标题数量要一致")
    readers = [cv2.VideoCapture(str(path)) for path in args.inputs]
    counts = [int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) for reader in readers]
    fps = readers[0].get(cv2.CAP_PROP_FPS) or 20.0
    if len(set(counts)) != 1 or counts[0] <= 0:
        raise RuntimeError(f"帧数不一致 {counts}")
    ok, sample = readers[0].read()
    if not ok:
        raise RuntimeError("第一帧读不出来")
    scale = args.column_width / sample.shape[1]
    column_h = int(round(sample.shape[0] * scale))
    readers[0].set(cv2.CAP_PROP_POS_FRAMES, 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (args.column_width * len(readers), column_h),
    )
    written = 0
    while True:
        frames = []
        finished = False
        for reader, label in zip(readers, args.labels):
            ok, frame = reader.read()
            if not ok:
                finished = True
                break
            resized = cv2.resize(frame, (args.column_width, column_h), interpolation=cv2.INTER_AREA)
            cv2.rectangle(resized, (0, 0), (220, 42), (0, 0, 0), -1)
            cv2.putText(resized, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            frames.append(resized)
        if finished:
            break
        writer.write(np.concatenate(frames, axis=1))
        written += 1
    writer.release()
    for reader in readers:
        reader.release()
    if written != counts[0]:
        raise RuntimeError(f"写出 {written} 帧 输入是 {counts[0]}")
    print(f"写出 {args.output} {written} 帧")


if __name__ == "__main__":
    main()
