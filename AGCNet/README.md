# AGCNet Reimplementation Settings

This directory documents the configuration and reproduction conditions used for the AGCNet baseline in the comparative experiments of the AST manuscript.

**Important:** this directory is not a standalone release of a complete AGCNet training/inference implementation. The complete reimplementation source code is not distributed here. Instead, the repository provides the paper-reported settings, the concrete settings used in our engineering reimplementation, the dataset split, and the implementation choices required to understand how the baseline was configured.

## Reference Method

**AGCNet: Improving Inertial Odometry via IMU Accelerometer and Gyroscope Online Compensation**  
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025.

AGCNet is used in our study as a source-end six-axis IMU compensation baseline. Its output is evaluated using the same downstream GNSS/IMU fusion protocol as the other source-end methods.

## What Is Provided Here

The main configuration file is:

```text
configs/uzh_fpv.yaml
```

It records:
1. **Paper-reported settings** directly supported by the published method description.
2. **Reproduction settings** actually used in our engineering reimplementation.
3. **Implementation choices** for details not fully specified in the original paper.

The complete training and inference source code used in our internal reimplementation is not included in this public repository.

## Dataset Split

### Training
```text
UZH_Indoorforward_3_davis
UZH_Indoorforward_7_davis
UZH_Outdoorforward_1_davis
UZH_Indoor45_2_davis
```

### Validation
```text
UZH_Indoorforward_5_davis
UZH_Outdoorforward_5_davis
UZH_Indoor45_4_davis
```

### Test
```text
UZH_Indoorforward_9_davis
UZH_Outdoorforward_3_davis
UZH_Indoor45_9_davis
UZH_Outdoor45_1_davis
```

## Paper-Reported Settings

| Item | Setting |
|---|---|
| Input window | 512 IMU samples |
| Prediction horizon | 10 samples |
| Input channels | 6 |
| Output channels | 6 |
| Backbone | Three-level dilated-convolution U-Net |
| Base channels | 32 |
| Encoder dilation rates | 1, 4, 16 |
| Dilated blocks per stage | 2 |
| Skip connections | Enabled |
| Accelerometer loss | Integrated-velocity Huber loss |
| Velocity Huber threshold | 0.05 |
| Gyroscope loss | SO(3)-integrated rotation Huber loss |
| Rotation Huber threshold | 0.005 |
| Tail weighting | N/T |
| Tail weight factor | 51.2 |
| Optimizer | Adam |
| Epochs | 1600 |
| Learning rate | 0.01 |
| Weight decay | 0.1 |
| Scheduler | Cosine annealing |

## Concrete Reproduction Settings

The concrete settings used in our engineering reproduction are preserved in `configs/uzh_fpv.yaml`, including:

```text
seed: 42
device: cuda
window: 512
horizon: 10
stride: 10
batch_size: 8
epochs: 1600
learning_rate: 0.01
weight_decay: 0.1
num_workers: 4
gravity: [0.0, 0.0, -9.81]
kernel_size: 5
dilations: [1, 4, 16]
```

## Implementation Choices Not Fully Specified by the Original Paper

The original paper does not fully specify every operator-level detail of Patch Merging, Patch Expanding, skip fusion, and the final projection layer.

Our engineering reproduction used:
- adjacent-frame concatenation followed by linear projection for patch merging;
- subchannel rearrangement for patch expanding;
- same-scale addition for skip fusion;
- a `64 -> 64 -> 6` projection head.

These items are explicitly separated under `implementation_choices` in the configuration file and must not be interpreted as unpublished settings supplied by the original AGCNet authors.

## Data Availability and Repository Paths

The evaluation files used by the downstream reproduction package are stored under the repository-level `data/` directory.

The configuration uses repository-relative paths:
```text
../data/raw
../data/gt
```

The files under `data/raw` and `data/gt` are processed evaluation files derived from the public UZH-FPV dataset and converted into the unified format used by our evaluation pipeline. Users should cite and comply with the license and terms of the original UZH-FPV dataset.

The complete original third-party dataset distribution is not reproduced here.

## Relation to the Released Table 3 Reproduction

The public repository's main executable reproducibility target is the downstream navigation evaluation.

The script:
```text
../reproduce_table3.py
```
reproduces the Baseline and Ours columns of Table 3 from the released frozen IMU outputs, outage seeds, and common 15-state ESKF evaluation protocol.

This AGCNet directory is provided to disclose the configuration and reproduction conditions of the AGCNet comparison. It is not required for executing the released Table 3 Baseline/Ours reproduction script.

## Scope

This directory is a **configuration and reproduction-conditions disclosure** for the AGCNet baseline. It is not an official AGCNet repository and does not claim bitwise equivalence to an unpublished original implementation.
