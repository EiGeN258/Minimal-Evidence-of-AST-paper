# LearnedBias 论文复现项目

对应论文：Abdullah Altawaitan 等，`Learned IMU Bias Prediction for Invariant Visual Inertial Odometry`，IEEE RA-L 2025。

本项目复现论文中可由正文确定的核心链路：200 Hz 六轴 IMU 的 1 秒滑窗、单网络联合估计陀螺仪与加速度计偏置、窗口内恒定六轴偏置、可微分 IMU 积分、`SE₂(3)` 姿态/速度/位置误差训练，以及在线逐帧补偿接口。输出可以直接接统一的惯导或 GNSS/IMU ESKF。

## 论文对应关系

| 项目 | 论文设定 | 本项目 |
|---|---:|---:|
| 输入 | 200 Hz IMU，1 秒 | `[B,200,6]` |
| 输出 | 六轴偏置 | `[B,200,6]`，窗口内恒定 |
| 最终网络 | ResNet，约 300K 参数 | 一维残差网络，默认约 30 万参数 |
| 状态损失 | `SE₂(3)` Huber | 姿态、速度、位置可微分积分损失 |
| 分量权重 | 姿态 `10³`，位置 `10²`，速度 `10¹` | 完整实现 |
| 优化器 | Adam，学习率 `1e-3` | 完整实现 |
| 推理 | 200 Hz 重叠滑窗 | `StreamingBiasCorrector` |

## 复现边界

论文没有公开最终 ResNet 的逐层通道、卷积核和残差块数量，只说明沿用所引 ResNet 设计且参数量约 300K。因此本项目使用满足论文输入、输出、恒定窗口偏置和参数规模约束的一维 ResNet。该部分属于工程复现，不声称与作者私有实现逐层相同。

论文的完整视觉前端、特征追踪和 MSCKF 工程细节不足以仅由正文无歧义复原。本项目实现论文的 LearnedBias 核心网络、训练损失和实时 IMU 补偿接口，不伪称复现作者完整视觉系统。用于你的统一 ESKF 对比时，这一边界正好保证所有方法只改变源端 IMU。


## 数据获取与许可说明

本仓库不分发复现实验所需的原始数据集。原始 IMU、轨迹真值及相关传感器数据应由使用者自行从相应公开数据集的官方来源下载，并遵守原数据集的许可协议、使用条款与引用要求。

仓库中的 `configs/` 目录及各序列目录仅用于提供实验配置和预期的数据组织结构；其中的空目录不代表数据缺失。下载原始数据后，请按照本 README 或配置文件中的路径要求放置数据，再执行预处理、训练和推理流程。

本仓库若提供模型生成的补偿后 IMU、预测偏置或评测结果，仅用于复现本文中的对比实验与统一下游评测，不构成对原始数据集的重新分发，也不改变原始数据集各自的版权与许可要求。

## 安装与自检

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -e ".[test]"
learned-bias-smoke
pytest
```

Linux 下将激活命令换为 `source .venv/bin/activate`。


输出 NPZ 包含：

- `time`: 相对秒
- `gyro`: rad/s，三轴
- `accel`: m/s²，三轴比力
- `position`: m
- `velocity`: m/s
- `quaternion`: `w,x,y,z`

建立 `data/train.txt` 与 `data/val.txt`，每行写一个相对该清单文件的 NPZ 路径。

## 训练

```bash
learned-bias-train --config configs/euroc.yaml
```

最佳权重输出到 `runs/learned_bias/best.pt`。

## 导出补偿 IMU

```bash
learned-bias-infer --checkpoint runs/learned_bias/best.pt --input data/V1_02_medium.npz --output outputs/V1_02_medium.csv
```

CSV 前七列为 `time + corrected gyro_xyz + corrected accel_xyz`，后六列为预测偏置。接入现有评测脚本时取前七列即可。

## 与当前 AirIMU 对比协议衔接

该项目只输出源端补偿后的六轴 IMU，不改你的伪 GNSS、失锁 seed、15-state ESKF、恢复策略和评价指标。由此可将 LearnedBias 与 Raw、AirIMU、DIDO、IMUDB 和 de-gyro 放进同一后端协议比较。
