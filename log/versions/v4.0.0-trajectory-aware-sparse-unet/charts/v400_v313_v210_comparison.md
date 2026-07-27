# v4.0.0 / v3.1.3 / v2.1.0 指标对比

说明：IoU、seg_acc、PD 越高越好，FA 越低越好。

| 版本 | 架构/变体 | IoU | seg_acc | PD | FA | max_events_num | 输入通道 | 提交校验 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v4.0.0 | Trajectory-Aware Sparse U-Net | 0.469880 | 0.846274 | 0.790424 | 0.000556096 | 50000 | 8 | ok / 24 txt |
| v3.1.3 | no-patchattention EV-SpSegNet | 0.717086 | 0.807605 | 0.809744 | 0.000152564 | 50000 | 8 | ok / 24 txt |
| v2.1.0 | v002_slope_0p005 improvement suite | 0.652904 | 0.810155 | 0.862873 | 0.000108761 | 20000 | 4 | ok / 24 txt |

## v4.0.0 相对差异

| 对照版本 | Delta IoU | Delta seg_acc | Delta PD | Delta FA |
|---|---:|---:|---:|---:|
| v3.1.3 - v4.0.0 | +0.247206 | -0.038668 | +0.019320 | -0.000403532 |
| v2.1.0 - v4.0.0 | +0.183024 | -0.036119 | +0.072449 | -0.000447335 |

## 结论

- v4.0.0 首版 Trajectory-Aware Sparse U-Net 架构闭环成功，已完成 Docker/conda 50 epochs 训练、chunked evaluate、submission 和 validate-submission。
- v4.0.0 使用 `input_channel=8`、`max_events_num=50000`，提交校验状态为 `ok`，24 个 txt 文件且错误列表为空。
- 指标上，v4.0.0 的 IoU=0.469880、PD=0.790424，低于 v3.1.3/v2.1.0；FA=0.000556096，明显高于 v3.1.3 的 0.000152564 和 v2.1.0 的 0.000108761。
- v4.0.0 的 seg_acc=0.846274 在本组三版本中最高，但该优势未转化为 IoU/PD/FA 综合优势。
- 后续 v4.1.0/T3 应聚焦提升轨迹完整度和压低虚警；v4.2.0/T4 与 v4.3.0/T5 顺延执行。
