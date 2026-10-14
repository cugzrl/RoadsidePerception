# R39_RAG

R39路口历史结构化数据的清洗、时空知识库和Key-Value运动检索实现。跟踪算法在上层 `algorithms/stmc`

## 目录

| 路径 | 用途 |
| --- | --- |
| `configs/` | 数据源、类别映射、字段约束、知识库参数 |
| `data/` | 原始链接、清洗后的序列、隔离样本 |
| `knowledge_base/` | 已构建的sqlite和npz，整理脚本时不覆盖 |
| `src/r39_rag/` | 清洗、投影、建库、检索和在线记忆的实现 |
| `scripts/data/` | 数据盘点、链接、清洗、校验 |
| `scripts/build/` | 时空知识库、Key-Value片段构建 |
| `scripts/query/` | 按位置或轨迹检索 |
| `scripts/experiments/` | 留出序列检索对比，以及跟踪消融启动脚本 |
| `scripts/diagnostics/` | 投影、BEV可信度和轨迹碎片检查 |
| `scripts/visualization/` | 知识库图和BEV视频拼接 |

## 具体实现的核心代码

- `src/r39_rag/cleaning/`：多版本JSON对齐到统一观测
- `src/r39_rag/projection/`：四路Homography地面投影和跨相机误差
- `src/r39_rag/knowledge/build.py`、`spatial.py`、`trajectory.py`、`store.py`：网格、典型轨迹和落盘
- `src/r39_rag/knowledge/retrieval.py`：空间区域和典型轨迹查询
- `src/r39_rag/knowledge/kv_motion.py`：1秒历史对应2秒后续运动
- `src/r39_rag/knowledge/online.py`：在线记忆使用的车道和轨迹先验
