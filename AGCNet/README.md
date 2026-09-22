# AGCNet 论文复现项目

对应论文：`AGCNet: Improving Inertial Odometry via IMU Accelerometer and Gyroscope Online Compensation`，IEEE/RSJ IROS 2025。

本项目复现论文的在线六轴 IMU 补偿主链路：512 帧输入、Patch Partition、卷积嵌入、三层扩张卷积 U-Net、跳跃连接、未来 10 帧补偿、多任务速度/姿态积分损失，以及按 10 帧步长运行的在线补偿器。

## 论文对应关系

| 项目 | 论文设定 | 本项目 |
|---|---:|---:|
| 输入窗口 | `N=512` | `[B,512,6]` |
| 输出 | 对齐偏移 `T=10` 的六轴加性补偿 | `[B,512,6]` |
| 基础通道 | `C=32` | 32 |
| 编码器扩张率 | `1,4,16` | `1,4,16` |
| 编解码深度 | 三层 U-Net | 三层 U-Net |
| 每层卷积 | Dilated Block ×2 | Dilated Block ×2 |
| 加速度任务 | 速度积分 Huber，阈值 0.05 | 完整实现 |
| 陀螺仪任务 | SO(3) 姿态积分 Huber，阈值 0.005 | 完整实现 |
| 尾部权重 | 最后 `T` 帧乘 `N/T` | 完整实现 |
| 训练 | Adam，1600 epoch，lr 0.01，wd 0.1，余弦退火 | 完整实现 |

## 复现边界

论文给出了结构图、各层通道、卷积核和扩张率，但没有说明 Patch Merging、Patch Expanding、跳跃融合和 Conv Projection 的全部算子细节。本项目按结构图的时序长度和通道变化实现：相邻帧拼接降采样、线性投影、子通道重排上采样、同尺度相加跳跃融合，以及 `64→64→6` 投影头。这些位置属于工程复现，不能宣称与作者未公开实现逐算子一致。

论文公式使用世界系角速度完成姿态递推。本项目保留这一正文写法，以便最大限度对齐论文，而不是擅自替换成常见的体系角速度右乘形式。


## 数据获取与许可说明

本仓库不分发原始数据集。复现实验所需的原始 IMU、轨迹真值及相关传感器数据应由使用者自行从相应公开数据集的官方来源下载，并遵守原数据集的许可协议、使用条款与引用要求。

仓库中的 `configs/` 目录仅提供实验所需的序列配置与路径模板。下载原始数据后，请按照配置文件或 README 中说明的目录结构放置数据，再执行预处理、训练与推理流程。

本仓库中若提供由原始数据进一步生成的模型输出、补偿后 IMU 或评测结果，其用途仅为复现本文中的对比与下游评测；这些文件不代表对原始数据集的再分发，也不改变原始数据集各自的版权与许可要求。

## 安装与自检

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -e ".[test]"
agcnet-smoke
pytest
```

Linux 下将激活命令换为 `source .venv/bin/activate`。

## EuRoC 数据预处理

```bash
agcnet-prepare-euroc /path/to/MH_01_easy data/MH_01_easy.npz
```

输出 NPZ 包含 `time`、`gyro`、`accel`、`position` 和 `quaternion`。四元数顺序为 `w,x,y,z`。


## 训练

```bash
agcnet-train --config configs/euroc.yaml
```

最佳权重输出到 `runs/agcnet/best.pt`。

## 导出补偿 IMU

```bash
agcnet-infer --checkpoint runs/agcnet/best.pt --input data/MH_04_difficult.npz --output outputs/MH_04_difficult.csv
```

CSV 前七列为 `time + corrected gyro_xyz + corrected accel_xyz`，后六列为网络加性补偿量。前 512 帧以及新预测尚未到达的时刻保持零补偿，这是因果部署所必需的热启动阶段。

## 与当前 AirIMU 对比协议衔接

该项目只输出源端补偿后的六轴 IMU。将导出 CSV 的前七列送入现有统一 15-state ESKF，保持伪 GNSS、失锁窗口、seed、恢复策略和所有指标不变，即可作为 AGCNet 源端基线。
