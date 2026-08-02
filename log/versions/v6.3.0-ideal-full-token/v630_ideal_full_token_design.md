# v6.3.0 — 理想全量 Transformer 设计文档

> **定位**：v6.3.0 不是可运行代码版本，而是在假设显存充足（≥ 24GB）的条件下，v6.2.0 全量 Token Transformer 的**完整理想实现参考**。本文档独立于实际代码，供后续算法分析、论文写作、或硬件升级后参考。

---

## 一、网络总览

```
┌────────────────────────────────────────────────────┐
│                 TrajectoryAwareSparseUNet           │
│                 (8-channel input)                  │
├────────────────────────────────────────────────────┤
│                                                    │
│  Input (8ch sparse voxel)                          │
│      │                                             │
│      ▼                                             │
│  GatedInputFusion ──► 24ch                         │
│      │                                             │
│      ▼                                             │
│  stage1 (2×MSTB) ──── 24ch                         │
│      │                                             │
│      ▼                                             │
│  down1 [2,2,2] ────── 24→40ch                      │
│      │                                             │
│      ▼                                             │
│  stage2 (2×MSTB) ──── 40ch                         │
│      │                                             │
│      ▼                                             │
│  down2 [2,2,2] ────── 40→64ch                      │
│      │                                             │
│      ▼                                             │
│  stage3 (2×MSTB) ──── 64ch                         │
│      │                                             │
│      ▼                                             │
│  down3 [2,2,4] ────── 64→96ch                      │
│      │                                             │
│      ▼                                             │
│  bottleneck_mstb ──── 96ch (~4500 voxels)          │
│      │                                             │
│      ▼                                             │
│  ┌─────────────────────────────┐                    │
│  │ BottleneckTokenTransformer  │ ← 核心模块        │
│  │ (理想：全量token, 4头, FF=2×)│                    │
│  │ (全局 all-to-all attention)  │                    │
│  └─────────────────────────────┘                    │
│      │                                             │
│      ▼                                             │
│  BottleneckDirectionAggregation (skipped, >2000)   │
│      │                                             │
│      ▼                                             │
│  decoder3 (InverseConv + skip3) ── 96→64ch         │
│      │                                             │
│      ▼                                             │
│  decoder2 (InverseConv + skip2) ── 64→40ch         │
│      │                                             │
│      ▼                                             │
│  decoder1 (InverseConv + skip1) ── 40→24ch         │
│      │                                             │
│      ▼                                             │
│  Linear(24, 1) + Sigmoid ──► per-voxel probability │
│                                                    │
│  辅助输出（训练时）:                                  │
│  DirectionPredictionHead ──► 3D direction vector    │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## 二、输入特征（8 通道，固定训练集归一化）

| 通道 | 名称 | 含义 | 归一化方式 |
|---:|---|---|---|
| 0 | `x` | 事件 x 坐标 | `(x - mean_x) / std_x`（训练集统计） |
| 1 | `y` | 事件 y 坐标 | `(y - mean_y) / std_y` |
| 2 | `t` | 事件时间戳 | `(t - mean_t) / std_t` |
| 3 | `p` | 事件极性（±1） | `(p - mean_p) / std_p` |
| 4 | `d` | 局部密度 | `(d - mean_d) / std_d` |
| 5 | `L*` | PCA 线性度 | `(L* - mean_L) / std_L` |
| 6 | `C` | 极性一致性 | `(C - mean_C) / std_C` |
| 7 | `T` | 局部时间连续性 | `(T - mean_T) / std_T` |

- 统计量来自**全部训练集事件**，存入 `cache/v630_input_stats.json`
- 每个样本输入前应用 `(x - mean) / (std + ε)`，验证/提交流程复用同一组统计
- 4 个手工特征构造方式见 `compute_handcrafted_features()`（每样本独立计算 `minmax01` 后复用 train stats 标准化）

---

## 三、主干模块详解

### 3.1 GatedInputFusion（输入融合）

```
输入: 8ch 稀疏体素
    ├── raw_mlp: Linear(8→24) + LeakyReLU → raw_proj (24ch)
    ├── geo_mlp: Linear(8→24) + LeakyReLU → geo_proj (24ch)
    └── gate: Sequential(
            Linear(48→24),  # concat[raw_proj, geo_proj]
            LeakyReLU,
            Linear(24→24),
            Sigmoid
        )
输出: 24ch = raw_proj * gate + geo_proj * (1 - gate)
```

- 两条路径分别处理原始事件特征和手工几何特征
- 门控融合允许网络自适应选择每条路径的贡献

### 3.2 MSTB（Multi-Scale Temporal Block）

```
输入: C channels
│
├── local: SubMConv3d(3×3×3) + BN + LeakyReLU
│
├── space:
│   ├── SubMConv3d(5×1×1) + BN + LeakyReLU   (空间X)
│   └── SubMConv3d(1×5×1) + BN + LeakyReLU   (空间Y)
│
├── temporal (可学习softmax融合):
│   ├── time1: SubMConv3d(1×1×5), dilation=1
│   ├── time2: SubMConv3d(1×1×5), dilation=2
│   └── time4: SubMConv3d(1×1×5), dilation=4
│   └── fused = softmax(weights)[0]*t1 + [1]*t2 + [2]*t4
│
├── concat[local, space, temporal_fused] → 3C ch
├── SubMConv3d(3C→C, 1×1×1)  (compress)
├── BN + LeakyReLU
└── + identity skip connection
输出: C channels
```

- 每个 stage 含 2 个 MSTB
- bottleneck 含 1 个 MSTB
- temporal 权重为可学习参数，初始 `[0, 0, 0]`

### 3.3 SparseDown / SparseUp

```
SparseDown(in→out, stride):
    SparseConv3d(in→out, 3×3×3, stride) + BN + LeakyReLU

SparseUp(in+skip→out):
    SparseInverseConv3d(in→out, 3×3×3)
    → concat[up_features, skip_features]
    → SubMConv3d(out+skip→out, 1×1×1) + BN + LeakyReLU
    → MSTB(out)
```

### 3.4 DirectionPredictionHead（训练辅助，推理禁用）

```
输入: bottleneck features (96ch)
结构: Linear(96→128) + LeakyReLU + Linear(128→3) + L2Normalize
输出: 3D 单位方向向量 (x, y, t)
```

- 仅在训练时参与损失 `L_dir = 0.2 * (1 - |cosine_similarity|)`
- 推理/提交时完全禁用

---

## 四、BottleneckTokenTransformer — 理想实现（核心）

### 4.1 初始化参数

```python
class BottleneckTokenTransformer(nn.Module):
    def __init__(self):
        channels = 96               # bottleneck 通道数
        token_limit = 8192          # 全量 (≈max voxels)
        num_heads = 4               # 多头注意力 (96/4=24 per head)
        ff_multiplier = 2           # FFN: 96 → 192 → 96
        dropout = 0.0
        num_layers = 1              # 单层 Transformer

        # 子模块
        self.coord_proj = Linear(3, 96)                  # 坐标投影
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=96,
                nhead=4,
                dim_feedforward=192,      # 96 * 2
                dropout=0.0,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=1,
        )
        self.out_norm = LayerNorm(96)                     # 输出归一化
        self.gate = Sequential(
            Linear(192, 96),                               # concat[orig, encoded]
            Sigmoid,
        )
```

### 4.2 参数量

| 模块 | 参数 | 计算 |
|---|---|---|
| `coord_proj` | Linear(3→96) | 3×96 + 96 = 384 |
| `encoder` (self-attn) | Q,K,V: 3 × Linear(96→96) | 3×(96×96+96) = 27,936 |
| `encoder` (out_proj) | Linear(96→96) | 96×96+96 = 9,312 |
| `encoder` (FF) | Linear(96→192) + Linear(192→96) | 96×192+192 + 192×96+96 = 37,056 |
| `encoder` (LN×2) | 2 × LayerNorm(96) | 2×(96+96) = 384 |
| `out_norm` | LayerNorm(96) | 192 |
| `gate` | Linear(192→96) | 192×96+96 = 18,528 |
| **总计** | | **≈ 93,792 参数** |

约 **94K 参数**，相比主干网络（数百万）非常轻量。

### 4.3 理想 forward 流程（无约束版）

```python
def forward(self, sp_tensor):
    features = sp_tensor.features   # [N, 96], N ≈ 4488
    indices = sp_tensor.indices     # [N, 4]  (batch, x, y, t)
    N = features.shape[0]

    # 1. 全量 token 选取 (token_limit >= N, 全部进入)
    token_idx = torch.arange(N)     # [N]

    # 2. Token 构造: 特征 + 坐标位置编码
    token_features = features[token_idx]           # [N, 96]
    coords = indices[token_idx, 1:].float()        # [N, 3]  (x,y,t)
    coords_norm = (coords - coords.min()) / (coords.max() - coords.min() + 1e-8)
    tokens = token_features + self.coord_proj(coords_norm)  # [N, 96], additive pos enc

    # 3. 全局 self-attention (无chunk, 无checkpoint)
    #    所有 N 个 token 一次性通过 TransformerEncoder
    encoded = self.encoder(tokens.unsqueeze(0)).squeeze(0)  # [1,N,96] → [N,96]
    encoded = self.out_norm(encoded)                         # [N,96]

    # 4. 门控残差回注
    gate = self.gate(torch.cat([token_features, encoded], dim=1))  # [N,96], sigmoid
    fused = token_features + gate * encoded                          # [N,96]

    # 5. 写回所有体素 (100% 覆盖)
    out_features = features.clone()
    out_features[token_idx] = fused

    # 6. 回到原始 tensor
    return sparse_replace(sp_tensor, out_features)
```

### 4.4 门控残差机制

```
fused = original + gate * encoded
          ↑            ↑
      恒等路径     Transformer增强
```

- `gate = Sigmoid(Linear(concat[original, encoded]→96))`
- 逐通道门控：每个维度独立决定接受多少 Transformer 增强
- 初始可接近恒等（如果 Transformer 输出无信息），训练中自适应激活
- 类似 LSTM/GRU 的门控思想，保证 Transformer 不会破坏有用的原始特征

---

## 五、损失函数（训练时）

```
L_total = L_main + 0.3 * L_traj + 0.2 * L_dir
```

| 项 | 权重 | 含义 | 计算方式 |
|---|---|---|---|
| `L_main` | 1.0 | 主分割损失 | `F.binary_cross_entropy(pred, label)` |
| `L_traj` | 0.3 | Dice 辅助损失 | `dice_loss(projected_pred, projected_label)`（XY 均值投影） |
| `L_dir` | 0.2 | 方向预测损失 | `1 - |cosine_similarity(pred_dir, gt_dir)|` |

- 仅训练时计算，推理只输出 `L_main`
- Dice 投影：沿 T 轴取 XY mean 到 BEV 面，计算 2D Dice

---

## 六、与 v6.2.0 实际实现的差异对比

| 项目 | v6.2.0（实际，7.6GB GPU） | v6.3.0（理想，≥24GB GPU） |
|---|---|---|
| **token 数量** | 全量 ~4488 | 全量 ~4488 |
| **token 覆盖率** | 100% | 100% |
| **注意力类型** | **chunked**（3块×1500，块间不交互） | **全局 all-to-all**（单次 forward） |
| **attention heads** | **2** | **4** |
| **FF multiplier** | **1**（dim_feedforward=96） | **2**（dim_feedforward=192） |
| **gradient checkpointing** | **有**（recompute forward in backward） | **无**（直接 backward） |
| **float32 cast** | features 显式 `.float()` 再 `.to(orig_dtype)` | 全程 autocast 无需手动 cast |
| **编码器每步操作** | 3 次 `encoder(chunk)` | 1 次 `encoder(tokens)` |
| **参数量** | ~53K（因 2 头+FF=1× 减半） | ~94K |
| **显存峰值** | ~6.5 GB（chunk+ckpt 控制） | ~14 GB（估计） |
| **每 epoch 时间** | ~1 分钟 | ~0.7 分钟（无 chunk overhead） |

### 为什么 v6.3.0 的 IoU 预期高于 v6.2.0

v6.2.0 的 chunked attention 将 4488 个 token 分成 3 组，每组内部做 self-attention，但**不同组的 token 之间没有任何交互**。这意味着：

- 组 A 的体素 B 的体素之间可以互相建模
- 但组 A 的体素 A 的体素 B 组内的体素 B 之间**完全看不到对方**

v6.3.0 的全局 attention 中，任意两个体素都能直接建立依赖：

```
v6.2.0 chunked attention matrix:     v6.3.0 global attention matrix:
┌──────┬──────┬──────┐               ┌─────────────────────┐
│  A   │      │      │               │                     │
│1500² │  0   │  0   │               │                     │
├──────┼──────┼──────┤               │      4488²          │
│  0   │  B   │      │               │    all-to-all       │
│      │1500² │  0   │               │                     │
├──────┼──────┼──────┤               │                     │
│  0   │  0   │  C   │               │                     │
│      │      │1488² │               │                     │
└──────┴──────┴──────┘               └─────────────────────┘
```

全局 attention 对 UAV 小目标分割尤其重要：一个目标的体素可能分散在瓶颈空间的任意位置，需要远距离交互才能建立完整上下文。

---

## 七、显存与计算量估算

### 7.1 注意力矩阵

```
tokens:               N = 4488
heads:                H = 4
head_dim:             D = 96/4 = 24
attention_matrix:     H × N × N × float32 = 4 × 4488 × 4488 × 4 bytes
                    = 322 MB (仅 attention logits)
attention_softmax:    同上 = 322 MB (需存储用于 backward)
```

### 7.2 前馈网络

```
FF hidden:            4488 × 192 × float32 = 3.4 MB
FF gate:              4488 × 192 × float32 = 3.4 MB
FF output:            4488 × 96 × float32 = 1.7 MB
```

### 7.3 总估算

| 阶段 | 显存 | 说明 |
|---|---|---|
| **主干网络**（不含 Transformer） | ~4.0 GB | encoder/decoder + MSTB + skip connections |
| **Transformer forward** | ~0.8 GB | tokens + encoded + gate activations |
| **Transformer backward**（保存的中间量） | ~1.5 GB | attention matrix(322MB) + Q/K/V(3×322MB) + FF intermed |
| **峰值总计** | **~14-16 GB** | v6.2.0 峰值 ~6.5GB；差值来自去掉 chunk/ckpt |

### 7.4 计算量

```
全局自注意力:           O(N² × D) = O(4488² × 96) ≈ 1.93 GFLOPs
FF 前馈网络:            O(N × D² × 4) = O(4488 × 96² × 4) ≈ 0.17 GFLOPs
positional encoding:    O(N × D) ≈ 0.0004 GFLOPs (可忽略)

总计: ≈ 2.1 GFLOPs per sample
```

对一个 50 epoch 的训练，每个 epoch 99 个 batch（train）+ 24 个 val，总量约为 `99 × 50 × 2.1 = 10.4 GFLOPs` 仅 Transformer 部分，相对于 3D 稀疏卷积的数百 GFLOPs 来说非常轻量。瓶颈在**显存带宽**而非计算。

---

## 八、训练配方

| 参数 | 值 |
|---|---|
| 优化器 | Adam |
| 学习率 | 0.001（constant） |
| Batch size | 1 |
| Epochs | 50 |
| Mixed precision | AMP（autocast） |
| 最大事件数 | 50,000 |
| 输入通道 | 8（固定训练集 z-score 归一化） |
| Seed | 316 |
| 推理分块 | 50,000（chunked inference） |
| 损失 | `BCE + 0.3*Dice + 0.2*Direction` |

---

## 九、推理与提交流程

### 9.1 推理模式

```python
# 禁用所有辅助模块
model.eval()

# 不传 return_direction=True
output, voxel = model(voxel_ev)

# output: [num_voxels, 1] 概率
# voxel: SparseConvTensor with probability features
```

### 9.2 事件级预测（p2v 映射）

```python
# p2v_map: 每个事件点 → 对应体素索引
voxel_probs = output[p2v_map]          # [num_events, 1]
event_labels = voxel_probs > threshold  # 二值化
```

### 9.3 提交格式

```
<val_000.txt>
x y t p label
128 64 0.00123 1 1
256 128 0.00456 -1 0
...

- x, y: 像素坐标 (整数)
- t: 归一化时间戳 (浮点)
- p: 极性 (+1 / -1)
- label: 0 (背景) / 1 (前景目标)
```

### 9.4 阈值选择策略

v6.2.0 的经验：
- best-IoU threshold = 0.70（最低阈值，IoU 最高）
- 阈值越低 → IoU 越高、FA 越高
- 阈值越高 → IoU 下降、FA 下降

v6.3.0 理想情况下预期 best-IoU threshold 会**上移到 0.75-0.85 区间**（因为全局 attention 带来更好的特征区分度）。

---

## 十、与 v6.0→v6.1→v6.2 的覆盖率进化线

| 版本 | Token 覆盖 | 注意力类型 | best-IoU | FA | 说明 |
|---|---:|---|---:|---:|---|
| v6.0.0 | 512/4488 (11%) | global（采样） | 0.485 | 0.001197 | 采样丢失空间信息 |
| v6.1.0 | 2048/4488 (46%) | global（采样） | 0.631 | 0.000459 | 覆盖率↑ → IoU+30% |
| v6.2.0 | 4488/4488 (100%) | **chunked**（块间不交互） | 0.765 | 0.000165 | 全量但非真正全局 |
| **v6.3.0** | **4488/4488 (100%)** | **global（真正 all-to-all）** | **预期 >0.780** | **预期更低** | 理想情况 |

**覆盖率→IoU 因果关系链全程单调成立**，v6.3.0 是这条线上的终点——真正实现了所有体素的全局注意力交互。

---

## 十一、已知限制与后续方向

1. **O(N²) 扩展性**：当体素数 >8000 时，即使 16GB GPU 也可能 OOM。后续可换用线性注意力（Performer / Linear Transformer）实现 O(N) 复杂度。
2. **单层 Transformer**：当前仅 1 层，表达能力有限。2-3 层可进一步建模高层语义依赖，但显存需求线性增加。
3. **位置编码增强**：当前仅用 additive Linear(coord→96)，可尝试 learned positional embeddings 或 sinusoidal encoding。
4. **跨尺度 Transformer**：可考虑在 stage3/bottleneck 多处插入，实现多尺度长程依赖。
5. **继续增大 token_limit**：当前 8192 已经足够 bneck，若未来 bottleneck 膨胀超过 8192 需同步上调。

---

> **文档版本**：v6.3.0（理想设计文档）  
> **编写日期**：2026-08-02  
> **基础版本**：v6.2.0（`log/versions/v6.2.0-full-token/`）  
> **此文档对应的实际运行版本**：v6.2.0（chunked attention, 2 头, FF=1×, checkpointing）  
> **本版本定位**：非可运行代码，仅供设计参考
