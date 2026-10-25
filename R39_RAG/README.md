# R39_RAG

把 R39 路口的历史结构化轨迹洗成统一坐标，建成时空知识库，供STMC全局跟踪在关联时检索

## 目录

| 路径 | 用途 |
| --- | --- |
| `configs/` | 数据源、类别、字段和知识库参数 |
| `data/` | 原始数据链接、清洗结果、隔离样本 |
| `knowledge_base/` | 已经建好的 sqlite、npz 和投影状态，不要覆盖 |
| `src/r39_rag/cleaning/` | 多版本 JSON 转成统一观测 |
| `src/r39_rag/projection/` | 检测框底边中点投影到地面，并统计跨相机误差 |
| `src/r39_rag/knowledge/` | 空间网格、典型轨迹、Key-Value 运动、可靠性先验、检索 |
| `scripts/data/` | 链接原始数据、清洗、校验 |
| `scripts/diagnostics/verify_bev.py` | 投影检查，写出 `projection_status.json` |
| `scripts/build/` | 建时空库、Key-Value 库、远端典型路径 |
| `scripts/query/` | 按位置或轨迹检索 |
| `scripts/experiments/eval_kv_retrieval.py` | 在留出序列上比较位置误差，只写数值结果 |
| `tests/` | 清洗、投影、检索和在线关联的单元测试 |

与全局跟踪STMC的接入在 `algorithms/stmc/src/tracker/knowledge_prior.py` 和 `motion_memory.py`。前者查历史空间和典型轨迹，后者用在线记忆、道路可达和Key-Value恢复中断轨迹。

## 建议阅读顺序

1. `configs/sources.yaml`、`configs/knowledge.yaml`：序列划分和投影、建库参数
2. `cleaning/convert.py`、`jobs.py`：一条历史 JSON 如何变成统一 JSONL
3. `projection/homography.py`、`projection/evaluate.py`：地面坐标和能否融合
4. `knowledge/build.py`、`spatial.py`、`trajectory.py`、`trust_prior.py`：空间知识和典型轨迹
5. `knowledge/kv_motion.py`：1 秒历史对应之后 2 秒运动
6. `knowledge/retrieval.py`、`knowledge/online.py`：检索结果如何变成关联里的有界修正
7. `algorithms/stmc/src/tracker/knowledge_prior.py`、`motion_memory.py`：上述结果在单阶段图关联中的调用位置

## 编码检索与跟踪量化评估（2026-10-25）

新增历史Transformer、场景MLP、FAISS编码库和多候选似然融合。新增TrackEval评估器，用两个留出序列的结构化结果作为伪真值，比较相同检测下的STMC和编码KV
