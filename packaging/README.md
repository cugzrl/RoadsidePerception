# 路侧感知系统部署说明

本系统用于离线处理 1～4 路视频，输出目标检测、跟踪、车牌识别结果和可视化视频。交付包已包含运行所需的环境、程序和模型，服务器准备好以下条件后即可部署。

## 一、部署前准备

| 项目 | 要求 |
|---|---|
| 操作系统 | Linux x86_64 |
| GPU | NVIDIA GPU；完整流程不支持纯 CPU 运行 |
| 宿主机软件 | Docker、NVIDIA 驱动、已配置的 NVIDIA Container Toolkit |
| 命令行工具 | Bash、tar、gzip、GNU coreutils |
| 磁盘空间 | 建议至少预留 30 GB 用于交付包、解压和镜像加载；输入视频和算法输出另计 |
| 已验证硬件 | RTX 4090 24 GB，驱动 580.178.04；其他硬件和驱动需先自检及短视频验收 |

**除上述服务器软件外，无需另外安装算法运行环境或下载模型。**

加载前，在准备运行任务的同一个终端确认以下命令执行成功：

```bash
docker info
nvidia-smi
```

这两条命令不能替代容器 GPU 自检。如果软件未安装或当前用户无 Docker 权限，请先由服务器管理员配置。宿主机配置完成后，镜像加载和算法运行均可离线完成。

## 二、解压并加载镜像（首次部署）

将 `roadside-perception-delivery-20260923.tar.gz` 放到服务器的可写目录中执行：

```bash
tar -xzf roadside-perception-delivery-20260923.tar.gz
cd roadside-perception-release-20260923
./load_image.sh
```

`load_image.sh` 自动执行：

1. 校验交付目录中所有文件的 SHA256。
2. 加载 `image.tar.gz` 中的 Docker 镜像。
3. 核对加载后的镜像 ID。

看到“镜像加载完成”表示成功。校验和加载大文件可能需要几分钟。**外层交付压缩包用于解压，里面的 `image.tar.gz` 才是 Docker 镜像归档**，不要将外层压缩包直接传给 `docker load`。

同一服务器加载成功后，无需每次运行都重复加载。

## 三、准备视频

以 `/data/roadside` 为工作目录示例；以下路径均可替换为当前用户有权限访问的实际路径。

```text
/data/roadside/
├── videos/                      # 只放本次需要一起处理的视频
│   ├── R39_Aw_CamS.mp4
│   ├── R39_Bn_CamW.mp4
│   ├── R39_Ce_CamN.mp4
│   └── R39_Ds_CamE.mp4
└── results/                     # 由脚本按需创建
```

- 视频目录第一层放入 1～4 个视频，不扫描子目录。支持 `.mp4`、`.mov`、`.mkv`、`.avi`、`.m4v`。
- 文件名去掉扩展名作为相机 ID；相机 ID 不可重复。可以只输入其中一路。
- 多路视频应属于同一场景，并已完成时间同步。目录内多个视频会作为一次多目任务处理，不是多个独立任务。
- 输入输出必须分开；每次正式任务建议使用新的输出目录。

### 使用内置 R39 标定

R39 场景对应的相机 ID 为 `R39_Aw_CamS`、`R39_Bn_CamW`、`R39_Ce_CamN`、`R39_Ds_CamE`，按上面的名称放置对应视频即可自动匹配。

### 使用其他相机标定

准备真实对应的标定文件，不能仅将其他场景的视频改名为 R39 相机名称。标定应与视频分辨率及拍摄场景一致，多目标跨相机关联所用的坐标系应一致。

```text
/data/roadside/calibrations/
├── camera_1/calibration.json
└── camera_2/calibration.json

/data/roadside/videos/
├── camera_1.mp4
└── camera_2.mp4
```

也支持在标定根目录直接放 `camera_1.json`、`camera_2.json`。标定 JSON 需含有效的 3×3 `homography` 矩阵。运行前在同一终端设置：

```bash
export ROAD_CALIBRATION_DIR=/data/roadside/calibrations
```

该设置仅在当前终端及其子进程中生效，重新登录后需重新设置。若切回内置标定，执行 `unset ROAD_CALIBRATION_DIR`。

## 四、自检并试运行

以下命令在解压后的交付目录执行。自检和试运行应使用与正式任务一致的 GPU、标定等配置。

```bash
./check_env.sh /data/roadside/videos /data/roadside/results/check
./run_container.sh /data/roadside/videos /data/roadside/results/smoke --max-frames 20
```

自检通过后，试运行按默认 20 FPS 对每路视频最多处理 20 帧，并执行完整算法。输入不足 20 帧时，以实际可提取帧数为准。

确认以下文件生成且可正常查看：

```text
/data/roadside/results/smoke/result.json
/data/roadside/results/smoke/visualization.mp4
```

### 自检检查什么

- 服务器平台、Docker 和指定 GPU 是否可用。
- 视频处理工具是否可用，必要的模型和字体文件是否存在。
- 跟踪组件能否执行 GPU 运算，车牌模型能否加载并完成一次测试推理。
- 指定输入目录中的视频数量、相机名称、视频流、标定矩阵格式。
- 输出目录的实际写入权限和剩余空间；不存在的输出目录会创建。

若只检查环境、不检查输入输出，可执行 `./check_env.sh`。自检通过后仍需完成短视频试运行，并检查实际结果；标定准确性、视频同步、识别效果和长视频资源需求应使用实际样本验证。

## 五、正式运行及结果

试运行完成后，使用新的输出目录执行正式任务：

```bash
set -o pipefail
./run_container.sh /data/roadside/videos /data/roadside/results/task_001 2>&1 | tee task_001.log
```

`set -o pipefail` 确保保存日志时仍能正确反映算法是否失败。看到“流水线完成”表示所有启用阶段已执行完成。视频较长时运行时间和磁盘占用会增加。

| 文件或目录 | 用途 |
|---|---|
| `result.json` | 最终结果，含逐帧检测框、类别、跟踪 ID、轨迹摘要和车牌结果 |
| `visualization.mp4` | 可视化视频，多目输入时拼接显示 |
| `manifest.json`、`detections.json` | 输入记录及检测中间结果 |
| `frames/`、`features/`、`tracking/` | 抽帧、特征和跟踪中间产物，排查或恢复任务时使用 |

主要查看前两个文件。JSON 中的文件路径为容器内路径；例如 `/data/output/frames/...` 对应本次宿主机输出目录下的 `frames/...`。空车牌不一定代表执行失败，应结合目标清晰度、大小、类别和跟踪关联判断。车牌仅对 Car、Van、Bus 类别处理。

运行过程无需联网，不修改输入视频，输出文件归当前运行用户所有。中间文件不会自动清理；需要续跑、排查问题或重新生成可视化时应保留。

## 六、常用配置

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `ROAD_GPU` | `0` | 选择单个宿主机 GPU 编号 |
| `ROAD_CALIBRATION_DIR` | 未设置 | 指定外部标定根目录 |
| `ROAD_SHM_SIZE` | `8g` | 容器共享内存上限，不是显存大小 |
| `ROAD_IMAGE` | 交付镜像标签 | 使用其他已加载的镜像 |

例如，改用宿主机第 1 号 GPU：

```bash
export ROAD_GPU=1
./check_env.sh
./run_container.sh /data/roadside/videos /data/roadside/results/task_002
```

脚本只映射所选的一张 GPU，容器内仍是 `cuda:0`，不要追加 `--device cuda:1`。

流水线参数放在输入和输出目录后面：

```bash
./run_container.sh /data/roadside/videos /data/roadside/results/task_003 \
  --det-batch-size 4 --feature-batch-size 32
```

其他参数说明：

- `--max-frames 200`：限制每路处理帧数，用于短序列验证。
- `--camera-ids ID1 ID2`：显式指定相机 ID，顺序对应视频文件名排序；此时可先不带目录参数做环境自检，再通过短视频验证匹配关系。
- `--overwrite`：允许重新抽帧覆盖已有任务内容；只在确认旧结果可以覆盖时使用。
- `--resume`：复用输出目录中已有的阶段结果，仅在输入和参数未改变、已有结果完整时使用。程序不会检查输入或参数是否变化，不要与 `--overwrite` 同用；输入或参数改变时使用新目录。

当前交付按 20 FPS 验证，请保持默认值。若直接传入文件路径参数，必须使用容器内路径；外部标定目录挂载到 `/data/calibrations`。

## 七、常见问题

| 现象 | 处理方法 |
|---|---|
| 解压失败或 SHA256 校验失败 | 检查传输是否完整、磁盘是否已满；重新获取交付包，不跳过校验 |
| Docker 无权限或无法连接 | 先确认当前用户可执行 `docker info`，由管理员处理服务或访问权限 |
| GPU runtime / CUDA 初始化失败 | 检查 NVIDIA 驱动、Container Toolkit 和 `ROAD_GPU`，保留自检日志 |
| 未找到相机标定 | 检查视频文件名与标定名称，确认 `ROAD_CALIBRATION_DIR` 在当前终端已设置 |
| 输出目录无权限 | 确认当前用户可写，避免复用其他用户创建的目录 |
| 抽帧目录非空 | 使用新输出目录；允许覆盖时才加 `--overwrite` |
| 显存不足 | 减小检测和特征批大小；跟踪阶段仍需满足自身显存需求 |
| 空间不足 | 分别检查 Docker 数据目录和输出目录所在磁盘，中间文件可能远大于原视频 |
| JSON 无车牌结果 | 检查实际车辆和车牌清晰度，结合运行日志判断，不等同于模型未加载 |

反馈问题时，请提供自检/运行日志、启动命令、GPU 型号和驱动版本，并注明收到的交付包名称。

## 八、交付文件说明

| 文件 | 说明 |
|---|---|
| `README.md` | 本部署说明 |
| `image.tar.gz` | 已包含算法运行环境、代码和权重的 Docker 镜像归档 |
| `load_image.sh` | 校验并加载镜像 |
| `check_env.sh`、`check_runtime.py` | 环境与输入输出自检 |
| `run_container.sh` | 启动完整算法流程 |
| `SHA256SUMS` | 解压后各文件的完整性校验清单 |


目录内其余记录文件供加载脚本或技术支持使用，无需手动操作。请保持交付文件完整，运行参数通过环境变量和命令行配置。
