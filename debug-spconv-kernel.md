# Debug Session: spconv-kernel

Status: OPEN

## Hypotheses
1. spconv 2.1.21 的 MaskImplicitGemm 预编译 CUDA kernel 未包含 RTX 2060 SUPER 的 sm_75，导致第一个 sparse conv 执行时报 no kernel image。
2. evspsegnet.py 中部分 SubMConv3d/SparseConv3d/SparseInverseConv3d 未显式指定 Native，默认算法落到 MaskImplicitGemm。
3. 只有 post_act_block 传入算法不足以覆盖 conv_input、SparseBasicBlock、patch_attention 等直接构造的 spconv 层。
4. 如果 Native 算法仍失败，则当前 spconv/cumm wheel 与 GPU 架构不匹配，需要用 CUMM_CUDA_ARCH_LIST=7.5/TORCH_CUDA_ARCH_LIST=7.5 从源码编译。

## Evidence
- 已定位所有 spconv 卷积构造点：model/evspsegnet.py、model/basemodel.py、utils/stcloss.py。
- 已将相关卷积显式设置为 spconv.ConvAlgo.Native，并同步到容器 /root/EV-UAV。
- 训练报错算法从 implicit_gemm_pair 变为 native_pair，但仍报 CUDA error: no kernel image is available for execution on the device。
- PyTorch CUDA 基础张量操作在 RTX 2060 SUPER sm_75 上正常，问题集中于 spconv/cumm 扩展 kernel。
- 容器 evuav 环境曾缺少 spconv，已安装 spconv-cu111==2.1.21 和 cumm-cu111==0.2.9；同时发现残留 editable cumm 0.8.2 元数据冲突。
- 卸载 cu111 wheel 后发现 easy-install.pth 中 /tmp/cumm 和 /tmp/spconv 旧路径抢占导入；已移除旧路径。
- 已 clone 精确版本 /tmp/cumm-029(v0.2.9) 和 /tmp/spconv-2121(v2.1.21)，并 editable 安装。
- 已手动构建 /tmp/cumm-029/cumm/core_cc，cumm 基础 pybind 扩展生成成功。

## Next Step
- 手动构建 /tmp/spconv-2121/spconv/core_cc。
- 运行最小 SubMConv3d CUDA 前向测试。
- 重新运行 python train.py 验证 batch 推进。
