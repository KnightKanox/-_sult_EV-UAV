# 传统非模型预测检查清单

## v1.1.0 第一版脚本可运行

- [x] 不依赖训练、不加载模型权重
- [x] 不依赖 torch、spconv、HAIS_OP
- [x] 使用 `numpy` 处理事件数组和指标统计
- [x] 连通域检测优先使用 OpenCV，缺失时可回退到 SciPy
- [x] 支持通过 `--data-root` 覆盖配置中的数据集路径
- [x] 支持 `--limit` 限制样本数，便于快速验证
- [x] 输出事件级 IoU、seg_acc/recall、precision、预测正事件数、GT 正事件数
- [x] `python traditional_baseline.py --help` 可在宿主机执行
- [x] `python traditional_baseline.py --data-root /root/EV-UAV-dataset --mode test --limit 1` 已在容器环境跑通
