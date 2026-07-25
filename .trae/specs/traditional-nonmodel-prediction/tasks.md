# 传统非模型预测任务

## v1.1.0 第一版跑通

- [x] 梳理 EV-UAV 当前配置、npz 字段说明和已有评估指标定义
- [x] 确认第一版方案采用事件密度阈值分割 + 二维连通域检测
- [x] 实现根目录脚本 `traditional_baseline.py`
- [x] 支持 `--config`、`--mode/--split`、`--data-root`、`--threshold`、`--min-area`、`--limit` 命令行参数
- [x] 从配置读取 `DATA.root` 和 `DATA.res`，并允许 `--data-root` 覆盖数据集路径
- [x] 遍历指定 split 下 `.npz` 文件，读取 `evs_norm`、`ev`/`ev_loc` 并映射到二维图像平面
- [x] 输出 IoU、seg_acc/recall、precision、预测正事件数、GT 正事件数和总事件数
- [x] 完成 `python traditional_baseline.py --help` 验证
- [x] 在容器数据集 `/root/EV-UAV-dataset` 上完成 `--mode test --limit 1` 验证
