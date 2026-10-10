"""SQLite 与 NPZ 读写"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = """
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE cells (
    cell_id INTEGER PRIMARY KEY,
    ix INTEGER NOT NULL,
    iy INTEGER NOT NULL,
    x_min REAL NOT NULL,
    y_min REAL NOT NULL,
    x_max REAL NOT NULL,
    y_max REAL NOT NULL,
    observation_count INTEGER NOT NULL,
    local_track_count INTEGER NOT NULL,
    clean_track_count INTEGER NOT NULL,
    global_track_count INTEGER NOT NULL,
    reassigned_track_count INTEGER NOT NULL,
    class_counts TEXT NOT NULL,
    direction_hist TEXT NOT NULL,
    speed_p50 REAL,
    speed_p90 REAL,
    camera_counts TEXT NOT NULL,
    observation_reliability TEXT NOT NULL,
    confidence TEXT NOT NULL,
    source_sequences TEXT NOT NULL
);
CREATE TABLE transitions (
    from_cell INTEGER NOT NULL,
    to_cell INTEGER NOT NULL,
    track_count INTEGER NOT NULL,
    median_dt_ms REAL,
    confidence TEXT NOT NULL,
    PRIMARY KEY (from_cell, to_cell)
);
CREATE TABLE prototypes (
    prototype_id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    support_tracks INTEGER NOT NULL,
    support_sequences TEXT NOT NULL,
    camera_counts TEXT NOT NULL,
    median_speed REAL,
    confidence TEXT NOT NULL,
    exemplar_sequence TEXT,
    exemplar_camera TEXT,
    exemplar_local_track_id INTEGER
);
CREATE TABLE cross_camera (
    camera_a TEXT NOT NULL,
    camera_b TEXT NOT NULL,
    relation TEXT NOT NULL,
    support_count INTEGER NOT NULL,
    median_distance_m REAL,
    median_dt_ms REAL,
    confidence TEXT NOT NULL,
    PRIMARY KEY (camera_a, camera_b, relation)
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def write_database(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        path.unlink()
    connection = connect(path)
    try:
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            [(key, json.dumps(value, ensure_ascii=False)) for key, value in payload["meta"].items()],
        )
        connection.executemany(
            "INSERT INTO cells VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            payload["cells"],
        )
        connection.executemany("INSERT INTO transitions VALUES (?, ?, ?, ?, ?)", payload["transitions"])
        connection.executemany(
            "INSERT INTO prototypes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            payload["prototypes"],
        )
        connection.executemany(
            "INSERT INTO cross_camera VALUES (?, ?, ?, ?, ?, ?, ?)",
            payload["cross_camera"],
        )
        connection.commit()
    finally:
        connection.close()


def write_arrays(path: Path, grid_count: np.ndarray, grid_tracks: np.ndarray, prototypes: np.ndarray, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        grid_count=grid_count,
        grid_tracks=grid_tracks,
        prototypes=prototypes,
        grid_min_x=float(config["grid_min_x"]),
        grid_min_y=float(config["grid_min_y"]),
        grid_size_m=float(config["grid_size_m"]),
    )


def load_arrays(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
