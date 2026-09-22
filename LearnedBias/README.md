# LearnedBias Reimplementation Settings

This directory documents the configuration and reproduction conditions used for the LearnedBias baseline in the comparative experiments of the AST manuscript.

**Important:** this directory is not a standalone release of a complete LearnedBias training/inference implementation. The complete reimplementation source code is not distributed here. Instead, the repository provides the paper-reported settings, the concrete settings used in our engineering reimplementation, the dataset split, and the implementation choices required to understand how the baseline was configured.

## Reference Method

**Learned IMU Bias Prediction for Invariant Visual Inertial Odometry**  
Abdullah Altawaitan et al., IEEE Robotics and Automation Letters, 2025.

In our comparative protocol, LearnedBias is used only as a source-end IMU bias-compensation method. Its corrected six-axis IMU output is evaluated with the same downstream GNSS/IMU fusion protocol as the other source-end methods.

## What Is Provided Here

The main configuration file is:

```text
configs/uzh_fpv.yaml
```

It records:
1. **Paper-reported settings** directly supported by the published method description.
2. **Reproduction settings** actually used in our engineering reimplementation.
3. **Implementation choices** for architectural details not fully disclosed in the original paper.

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
| IMU sampling rate | 200 Hz |
| Input window | 1 s |
| Input samples | 200 |
| Input channels | 6 |
| Predicted quantity | Six-axis IMU bias |
| Bias assumption | Constant within each window |
| Backbone | ResNet |
| Approximate parameter count | 300 K |
| Training objective | Differentiable IMU integration |
| State representation | SE2(3) |
| Robust loss | Huber |
| Orientation weight | 1000 |
| Position weight | 100 |
| Velocity weight | 10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Inference | Sliding-window bias correction |

## Concrete Reproduction Settings

The concrete settings used in our engineering reproduction are preserved in `configs/uzh_fpv.yaml`, including:

```text
seed: 42
device: cuda
window: 200
stride: 200
batch_size: 32
epochs: 200
learning_rate: 0.001
weight_decay: 0.0
num_workers: 4
gravity: [0.0, 0.0, -9.81]
channels: 96
kernel_size: 5
dilations: [1, 2, 4]
dropout: 0.0
huber_delta: 1.0
```

## Implementation Choices Not Fully Specified by the Original Paper

The original paper does not disclose the final ResNet layer-by-layer channel widths, exact kernel sizes, or complete residual-block count.

Our engineering reproduction therefore uses a one-dimensional residual network configured with:
```text
channels: 96
kernel_size: 5
dilations: [1, 2, 4]
dropout: 0.0
```

These entries are explicitly separated under `implementation_choices` in the configuration file and must not be interpreted as unpublished settings supplied by the original LearnedBias authors.

The original method also contains a broader visual-inertial estimation system. Our comparison uses only the learned IMU-bias compensation component so that all source-end methods are evaluated using the same downstream navigation backend.

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

This LearnedBias directory is provided to disclose the configuration and reproduction conditions of the LearnedBias comparison. It is not required for executing the released Table 3 Baseline/Ours reproduction script.

## Scope

This directory is a **configuration and reproduction-conditions disclosure** for the LearnedBias baseline. It is not an official LearnedBias repository and does not claim bitwise equivalence to an unpublished original implementation.
