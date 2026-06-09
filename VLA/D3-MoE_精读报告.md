# D3-MoE 论文精读报告

> **论文**: D$^3$-MoE: Dual Disentangled Diffusion Mixture-of-Experts for Style-Controllable End-to-End Autonomous Driving
> **arXiv**: [2606.04884](https://arxiv.org/abs/2606.04884) [cs.RO]
> **作者**: Renju Feng, Rukang Wang, Ning Xi, Jianguo Yu, Liping Lu, Pan Zhou, Duanfeng Chu
> **机构**: Wuhan University of Technology; Huazhong University of Science and Technology
> **日期**: 2026-06-03（v1）
> **代码**: [待验证] arXiv 页面未列出官方代码链接

---

## 0. 引用信息表

| 信息项 | 内容 |
|--------|------|
| 论文标题 | D$^3$-MoE: Dual Disentangled Diffusion Mixture-of-Experts for Style-Controllable End-to-End Autonomous Driving |
| arXiv ID | 2606.04884 [cs.RO] |
| 方法全称 | D$^3$-MoE: Dual Disentangled Diffusion Mixture-of-Experts |
| 任务领域 | 端到端自动驾驶、风格可控轨迹规划、Diffusion Planner、Mixture-of-Experts |
| 核心问题 | 高方差人类驾驶示范导致端到端 planner 出现 style-averaging，生成同质化、不可控、甚至运动学不安全的轨迹 |
| Benchmark | NAVSIM navtest |
| 关键结果 | Normal: 88.2 PDMS / 84.3 EPDMS；Best-of-Three: 91.3 PDMS / 87.5 EPDMS |

---

## 1. Motivation（问题背景）

### 1.1 端到端自动驾驶里的 style-averaging

端到端自动驾驶从多传感器观察直接预测规划轨迹，避免了传统模块化链路中感知、预测、规划之间的误差级联。但论文指出，主流 imitation learning planner 在高方差人类驾驶数据上存在一个结构性问题：**相似场景可能对应多种合理驾驶动作，而离线数据通常只记录一个示范轨迹**。

如果用回归式损失训练单一轨迹输出，模型在数学上会趋向条件均值。这个均值不一定是任何真实驾驶风格，可能变成介于“让行”和“超车”之间的半承诺轨迹，这就是论文称为的 style-averaging。

### 1.2 Related Works 与本文要解决的问题

| 方向 | 代表工作 | 局限 |
|------|----------|------|
| 统一端到端规划 | [UniAD](https://arxiv.org/abs/2212.10156), [TransFuser](https://arxiv.org/abs/2205.15997), [BEVFormer](https://arxiv.org/abs/2203.17270) | 强化了端到端表征，但不直接解决单示范监督下的风格平均 |
| 扩散式轨迹生成 | [DiffusionDrive](https://arxiv.org/abs/2411.15139), [GuideFlow](https://arxiv.org/abs/2506.03586) | 可以保留多模态，但标准随机采样缺少显式风格接口和确定性选择机制 |
| MoE 规划器 | ARTEMIS、TrajMoE、Geminus、MoSE 等 | 隐式 routing 缺乏物理语义；依赖人工场景标签的 routing 又难泛化到开放长尾场景 |

D$^3$-MoE 的动机是同时解决两个瓶颈：

1. **行为轴**：把“生成候选轨迹”和“选择最终轨迹”解耦，显式生成 aggressive / normal / conservative 多风格候选。
2. **物理轴**：把轨迹生成拆成横向与纵向两个正交维度，用自监督 routing 激活对应专家，而不是依赖人工场景标签。

> **图 1：Comparison of MoE architectures. (a) MoE models with implicit routing supervision inherently lack physical interpretability. (b) MoE models relying on explicit routing via manual scenario labels struggle to generalize across open-world environments. (c) Our proposed D$^3$-MoE framework compresses unbounded scenarios into combinations of finite, orthogonal physical primitives by explicitly decoupling lateral and longitudinal generation. Supervised purely by ground-truth kinematics rather than manual labels, independent routers activate specialized Diffusion Transformer (DiT) experts during inference. Equipped with style-conditioned AdaLN and asymmetric lateral-fusion cross-attention, these experts independently denoise their assigned physical states, which are subsequently reassembled to enable multi-style, kinematically coherent trajectory generation.**（对应论文 Figure 1）
>
> ![D3-MoE MoE 架构对比](./imgs/d3moe_fig1_moe_comparison.png)
>
> - **隐式 MoE**：专家会分工，但分工不一定对应可解释的物理动作。
> - **人工标签 MoE**：专家语义清楚，但依赖离散场景标签，开放世界泛化差。
> - **D$^3$-MoE**：把无限场景压缩为有限横向/纵向物理基元组合，并用轨迹运动学自监督 routing。

---

## 2. 一句话总结

D$^3$-MoE 用“双重解耦”的扩散 MoE 做风格可控自动驾驶规划：行为上并行生成多风格候选再选择，物理上把横向与纵向轨迹分给独立路由和 DiT 专家生成，从而缓解 style-averaging，并在 NAVSIM navtest 上达到 88.2 PDMS，Best-of-Three 进一步达到 91.3 PDMS。

---

## 3. 核心贡献

1. **提出 Dual Disentangled 设计**：行为轴解耦 generation/selection，物理轴解耦 lateral/longitudinal trajectory generation。
2. **提出自监督可解释 routing**：从 GT 轨迹的横向位移、航向变化、平均速度、平均加速度中构造软标签，避免人工场景标签。
3. **将 MoE 与 diffusion planner 结合**：横向 5 个专家、纵向 3 个专家，专家由 DiT 架构实现，并通过 style-conditioned AdaLN 注入驾驶风格。
4. **引入非对称横向融合**：纵向专家通过 lateral-fusion cross-attention 读取横向 hidden states，使速度/加速度与转向几何一致。
5. **实验证明多风格候选空间有上界收益**：Normal 单模型 88.2 PDMS，Best-of-Three 从三种风格中择优后提升到 91.3 PDMS。

---

## 4. 方法详述

### 4.1 问题定义

给定 ego-centric 坐标下的未来轨迹：

$$
\tau = \{(x_k, y_k, \theta_k)\}_{k=1}^K
$$

D$^3$-MoE 将其正交拆成纵向位移序列与横向/航向序列：

$$
\tau_{lon} = \{\Delta x_k\}_{k=1}^K
$$

$$
\tau_{lat} = \{(\Delta y_k, \Delta\theta_k)\}_{k=1}^K
$$

横向有 5 个专家：sharp left、left、straight、right、sharp right。纵向有 3 个专家：braking、cruising、accelerating。每次 forward 时，横向 router 和纵向 router 各自 Top-1 激活专家。论文采用 $x_0$-prediction，而不是标准 $\epsilon$-prediction：专家直接回归 clean trajectory component。

### 4.2 整体 Pipeline

```text
NAVSIM 多视角图像 + ego-state
        │
        ▼
VoVNetV2-99-eSE + FPN
        │
        ├── 多尺度视觉序列 F_vision
        └── 全局场景特征 F_global
        │
        ▼
Style Encoder + Ego Encoder
        │
        ▼
State-fusion Transformer
        │
        ▼
Decoupled Routers
        ├── lateral router: 5-way expert logits
        └── longitudinal router: 3-way expert logits
        │
        ▼
Style-aware DiT Experts
        ├── lateral denoiser: tau_lat
        └── longitudinal denoiser: tau_lon
             ▲
             └── asymmetric lateral-fusion cross-attention
        │
        ▼
Reassemble tau_hat
        │
        ▼
User-centric selection / Score-centric Best-of-Three selection
```

> **图 2：Methodological pipeline of D$^3$-MoE. Left: Offline multi-style expert reference trajectory augmentation. Middle: The core denoising architecture with dynamic routing and independent lateral/longitudinal DiT experts. Right: The behavioral decoupling paradigm for parallel multi-style trajectory generation and their subsequent downstream selection, which features a user-centric mode driven by language instructions or style indices, and a score-centric mode that generates parallel multi-style candidates for downstream evaluation and optimal selection.**（对应论文 Figure 2）
>
> ![D3-MoE 方法框架](./imgs/d3moe_fig2_pipeline.png)
>
> - **左侧**：离线合成 aggressive / normal / conservative 三风格参考轨迹。
> - **中间**：横向与纵向 router 分别激活 DiT experts，专家在 diffusion denoising 中预测对应物理分量。
> - **右侧**：输出多风格候选，既可由用户风格偏好选择，也可由 evaluator 做 Best-of-Three。

### 4.3 Dynamic Routing

输入图像 $\mathbf{I} \in \mathbb{R}^{3 \times H \times W}$ 经过 backbone + FPN 后得到视觉序列：

$$
\mathcal{F}_\text{vision} \in \mathbb{R}^{N_\text{vis} \times D_\text{context}}
$$

全局平均池化得到：

$$
\mathbf{F}_\text{global} \in \mathbb{R}^{D_\text{context}}
$$

当前 ego-state 被 MLP 编码为：

$$
\mathbf{e} \in \mathbb{R}^{D_\text{context}}
$$

style encoder 结合 $\mathbf{F}_\text{global}$ 与 $\mathbf{e}$ 生成第 $i$ 个 style embedding $\mathbf{s}_i$。State-fusion Transformer 处理 $[\mathbf{e}, \mathbf{s}_i, \text{cls}]$ 得到 query：

$$
\mathbf{Q}_{\text{state}, i}
$$

跨注意力从特征金字塔第 $j \in \{3, 6\}$ 层抽取空间语义上下文：

$$
\mathbf{z}_{i,j} = \text{CrossAttn}(\mathbf{Q}_{\text{state}, i}, \mathcal{F}_\text{vision}^{(j)})
$$

聚合上下文：

$$
\mathbf{Z}_i = [\mathbf{z}_{i,3}, \mathbf{z}_{i,6}]
$$

横向与纵向 routing logits 分别为：

$$
\mathbf{O}_{\text{lat}, i} = \Phi_\text{lat}(\mathbf{Z}_i), \qquad \mathbf{O}_{\text{lon}, i} = \Phi_\text{lon}(\mathbf{Z}_i)
$$

### 4.4 自监督 Routing Labels

论文从 GT 轨迹中提取四维运动学特征 $\mathcal{P}$：

| 轴 | 特征 | 含义 |
|----|------|------|
| lateral | $\Delta\theta$ | 净航向变化 |
| lateral | $d_{lat}$ | 有符号最大横向位移 |
| longitudinal | $\bar{v}$ | 平均速度 |
| longitudinal | $\bar{a}$ | 平均加速度 |

对标准化特征 $\mathcal{P}'$ 分别在横向/纵向维度做 K-means，得到 cluster center $\mathbf{C}_k$ 与标准差 $\boldsymbol{\sigma}_k$。软路由标签用 Mahalanobis-style softmax 计算：

$$
\hat{y}_k =
\frac{
\exp\!\left(
-\tfrac{1}{2\kappa}
\sum_{d}
\dfrac{(\mathcal{P}'_d - C_{k,d})^2}{\sigma_{k,d}^2}
\right)
}{
\sum_{j}
\exp\!\left(
-\tfrac{1}{2\kappa}
\sum_{d}
\dfrac{(\mathcal{P}'_d - C_{j,d})^2}{\sigma_{j,d}^2}
\right)
}
$$

Routing loss 是预测门控概率 $\mathbf{G}_a$ 与 physics-derived label 的 KL：

$$
\mathcal{L}_\text{route}
=
\sum_{a \in \{\text{lat},\, \text{lon}\}}
D_\text{KL}\!\left(\hat{\mathbf{y}}_a \,\big\|\, \mathbf{G}_a\right)
$$

### 4.5 Style-Aware MoE Diffusion

在扩散 timestep $t$，横向和纵向 forward process 得到：

$$
\tau_{\text{lat}, t}, \qquad \tau_{\text{lon}, t}
$$

每个专家由 $L$ 层 DiT block 组成，包含 Self-Attention、Multi-Scale Visual Cross-Attention、Unified Condition Cross-Attention 和 SwiGLU MLP。Unified condition 来自 ego-state 与 style feature 的投影：

$$
\mathbf{C}_\text{unified}
=
[\,\tilde{\mathbf{e}};\; \tilde{\mathbf{s}}_i\,]
\in
\mathbb{R}^{2 \times D_\text{context}}
$$

为了让纵向速度/加速度与横向路径几何一致，纵向专家在 MLP 前加入 asymmetric lateral-fusion cross-attention，只读取横向专家对应层的 hidden states：

$$
\mathbf{H}_\text{lat}^{(l)}
$$

Style 条件通过 AdaLN 注入。全局风格向量 $\mathbf{s}_\text{global}$ 与 sinusoidal timestep embedding $\mathbf{t}_\text{emb}$ 拼接后，经两层 SiLU MLP 输出每层 9 个调制参数：scale $\gamma$、shift $\beta$、residual gate $\alpha$，分别作用于 3 个子模块。

### 4.6 训练目标

专家分量损失使用 layer-wise deep supervision：

$$
\mathcal{L}_\text{expert}
=
\sum_{l=1}^{L}
w_l
\left(
\| \hat{\tau}_\text{lat}^{(l)} - \tau_\text{lat} \|_1
+
\| \hat{\tau}_\text{lon}^{(l)} - \tau_\text{lon} \|_1
\right)
$$

最终重组轨迹 $\hat{\tau}$ 用 hybrid $L_1$-$L_2$ reconstruction loss：

$$
\mathcal{L}_\text{traj}
=
\lambda_\text{L1}
\| \hat{\tau} - \tau_\text{GT} \|_1
+
\lambda_\text{L2}
\| \hat{\tau} - \tau_\text{GT} \|_2^2
$$

总目标为：

$$
\mathcal{L}_\text{total}
=
\mathbb{E}_{\tau_\text{GT}}
\Big[
\lambda_\text{route} \mathcal{L}_\text{route}
+
\mathbb{E}_{t, \epsilon}
\big[
\lambda_\text{traj} \mathcal{L}_\text{traj}
+
\lambda_\text{expert} \mathcal{L}_\text{expert}
\big]
\Big]
$$

其中 diffusion 部分对 $t \sim \mathcal{U}(1, T)$ 与 $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 取期望；routing loss 只由 GT kinematics 监督。

---

## 5. 训练与推理伪代码

### 5.1 训练流程

```python
def train_d3moe(model, navsim_batch):
    images = navsim_batch.multi_view_images
    ego_state = navsim_batch.ego_state
    gt_traj = navsim_batch.future_trajectory

    # 1. 将 GT 轨迹分解为横向与纵向物理分量
    tau_lon = extract_longitudinal(gt_traj)      # {Delta x_k}
    tau_lat = extract_lateral_heading(gt_traj)   # {(Delta y_k, Delta theta_k)}

    # 2. 从 GT 运动学中构造 self-supervised routing labels
    kinematic_features = compute_features(
        gt_traj,
        features=["net_heading_change", "signed_max_lateral_displacement",
                  "mean_speed", "mean_acceleration"],
    )
    y_lat, y_lon = mahalanobis_soft_labels(kinematic_features)

    # 3. 采样扩散时间步与噪声
    t = uniform_sample(1, model.num_diffusion_steps)
    eps_lat, eps_lon = sample_gaussian_like(tau_lat), sample_gaussian_like(tau_lon)
    noisy_lat = q_sample_x0(tau_lat, t, eps_lat)
    noisy_lon = q_sample_x0(tau_lon, t, eps_lon)

    # 4. 路由网络预测横向/纵向专家概率
    visual_tokens, global_feature = model.encode_scene(images)
    style_emb = model.encode_style(global_feature, ego_state)
    g_lat, g_lon = model.route(visual_tokens, ego_state, style_emb)

    # 5. Top-1 激活专家，分别做 x0-prediction
    lat_expert = top1(g_lat)
    lon_expert = top1(g_lon)
    pred_lat_layers = model.lateral_experts[lat_expert](
        noisy_lat, t, visual_tokens, ego_state, style_emb
    )
    pred_lon_layers = model.longitudinal_experts[lon_expert](
        noisy_lon, t, visual_tokens, ego_state, style_emb,
        lateral_hidden_states=pred_lat_layers.hidden_states,
    )

    # 6. 重组轨迹并优化三类损失
    pred_traj = reassemble(pred_lon_layers[-1], pred_lat_layers[-1])
    loss_route = kl_div(y_lat, g_lat) + kl_div(y_lon, g_lon)
    loss_expert = layerwise_l1(pred_lat_layers, tau_lat) + layerwise_l1(pred_lon_layers, tau_lon)
    loss_traj = lambda_l1 * l1(pred_traj, gt_traj) + lambda_l2 * l2_squared(pred_traj, gt_traj)
    loss = lambda_route * loss_route + lambda_expert * loss_expert + lambda_traj * loss_traj

    loss.backward()
    optimizer.step()
```

### 5.2 推理流程

```python
def infer_d3moe(model, images, ego_state, selection_mode="best_of_three"):
    candidates = []

    for style in ["aggressive", "normal", "conservative"]:
        visual_tokens, global_feature = model.encode_scene(images)
        style_emb = model.encode_style(global_feature, ego_state, style=style)

        g_lat, g_lon = model.route(visual_tokens, ego_state, style_emb)
        lat_expert = top1(g_lat)
        lon_expert = top1(g_lon)

        noisy_lat = sample_standard_normal(shape=model.lat_shape)
        noisy_lon = sample_standard_normal(shape=model.lon_shape)

        for t in reversed(model.denoising_schedule):  # paper default: 2 steps
            pred_lat, lat_hidden = model.lateral_experts[lat_expert].denoise(
                noisy_lat, t, visual_tokens, ego_state, style_emb
            )
            pred_lon = model.longitudinal_experts[lon_expert].denoise(
                noisy_lon, t, visual_tokens, ego_state, style_emb,
                lateral_hidden_states=lat_hidden,
            )
            noisy_lat, noisy_lon = diffusion_update(pred_lat, pred_lon, t)

        traj = reassemble(pred_lon, pred_lat)
        candidates.append({"style": style, "trajectory": traj})

    if selection_mode == "user_centric":
        return select_by_user_preference(candidates)

    if selection_mode == "best_of_three":
        scores = evaluator_score(candidates)
        return candidates[argmax(scores)]

    return candidates
```

---

## 6. 实验结论

### 6.1 数据集与训练设置

论文在 NAVSIM 上训练和评估。每个样本包含 8 路相机、5 个 LiDAR 融合点云、地图标注和 3D box。给定前 2 秒的 4 帧历史，模型预测未来 4 秒的 8 个 waypoint。

多风格轨迹增强如下：

| 风格 | 构造方式 | 作用 |
|------|----------|------|
| Normal | 直接使用 GT 轨迹 | 作为人类驾驶示范基线 |
| Aggressive | 将 GT 速度放大 1.05-1.25 倍，并沿 ego-lane centerline 重新延展 | 提供更主动的效率偏好 |
| Conservative | 降速、增大跟车间距，必要时强制减速和车道居中，最终可截断 planning horizon | 提供更谨慎的安全偏好 |

> **图 3：Distributions of the stylized reference trajectories. (a) Spatial density heatmaps of the different stylic trajectories. (b) Probability density distributions for speed, acceleration, and jerk. Aggressive trajectories shift toward higher speed, acceleration, and jerk, while Conservative ones concentrate at lower magnitudes, confirming kinematically distinct styles.**（对应论文 Figure 3）
>
> ![D3-MoE 三风格参考轨迹分布](./imgs/d3moe_fig3_style_distribution.png)
>
> - **Aggressive**：速度、加速度、jerk 分布整体右移，体现更主动的纵向推进。
> - **Conservative**：动力学量更低，说明合成策略确实形成谨慎驾驶偏好。
> - **Normal**：位于两者中间，保留数据集中人类示范的均衡模式。

### 6.2 实现细节

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Lateral Experts | $E_{\text{lat}}$ | 5 |
| Longitudinal Experts | $E_{\text{lon}}$ | 3 |
| Active Experts (Top-1) | $K_{\text{lat}}, K_{\text{lon}}$ | 1 |
| Training Epochs | - | 100 |
| Batch Size | - | 32 |
| Initial Learning Rate | - | $1 \times 10^{-4}$ |
| Weight Decay | - | $1 \times 10^{-4}$ |
| Diffusion Denoising Steps | - | 2 |
| Layer Weights | $w_1, w_2, w_3$ | 0.3, 0.65, 1.0 |
| L1 / L2 Loss | $\lambda_{L1}$ / $\lambda_{L2}$ | 10 / 5 |
| Routing Loss | $\lambda_{\text{route}}$ | 10 |
| Expert Loss | $\lambda_{\text{expert}}$ | 20 |
| Trajectory Loss | $\lambda_{\text{traj}}$ | 1 |

### 6.3 主实验：Base Metrics on Navtest

| Method | Input | NC | DAC | EP | TTC | C | PDMS |
|--------|-------|----|-----|----|-----|---|------|
| VADv2 | C&L | 97.2 | 89.1 | 76.0 | 91.6 | 100 | 80.9 |
| UniAD | C | 97.8 | 91.9 | 78.8 | 92.9 | 100 | 83.4 |
| Transfuser | C&L | 97.7 | 92.8 | 79.2 | 92.8 | 100 | 84.0 |
| PARA-Drive | C | 97.9 | 92.4 | 79.3 | 93.0 | 99.8 | 84.0 |
| DriveX | C | 97.5 | 94.0 | 79.7 | 93.0 | 100 | 84.5 |
| LAW | C | 96.4 | 95.4 | 81.7 | 88.7 | 99.9 | 84.6 |
| FSDrive | C | 98.2 | 93.8 | 80.1 | 93.3 | 99.9 | 85.1 |
| NoRD | C&L | 97.6 | 94.9 | 79.3 | 93.5 | 100 | 85.6 |
| Epona | C | 97.9 | 95.1 | 80.4 | 93.8 | 99.9 | 86.2 |
| DistillDrive | C&L | 98.1 | 94.6 | 81.0 | 93.6 | 100 | 86.2 |
| ARTEMIS | C&L | 98.3 | 95.1 | 81.4 | 94.2 | 100 | 86.9 |
| PRIX | C | 98.1 | 96.3 | 82.3 | 94.1 | 100 | 87.8 |
| DiffusionDrive | C&L | 98.2 | 96.2 | 82.2 | 94.7 | 100 | 88.1 |
| **D$^3$-MoE (Normal)** | C | **98.3** | **96.4** | **82.7** | 94.3 | 99.9 | **88.2** |
| D$^3$-MoE (Conservative) | C | 98.0 | 94.7 | 79.6 | 93.1 | 100 | 85.5 |
| D$^3$-MoE (Aggressive) | C | 96.1 | 91.2 | 79.5 | 87.1 | 99.9 | 80.9 |
| **D$^3$-MoE (Best-of-Three)** | C | 98.7 | 97.7 | 87.6 | 95.3 | 100 | **91.3** |

结论：Normal 风格单分支已经超过 DiffusionDrive 0.1 PDMS；Best-of-Three 利用三风格候选上界，把 PDMS 提升到 91.3。

### 6.4 Extended Metrics on Navtest

| Model | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
|-------|----|-----|-----|-----|----|-----|----|----|----|-------|
| Ego Status | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| Hydra-MDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | 89.1 | 83.1 |
| ReCogDrive | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 |
| iPad | 98.7 | 97.8 | 99.1 | 99.8 | 83.5 | 98.0 | 96.2 | 98.1 | 85.6 | 84.1 |
| **D$^3$-MoE (Normal)** | 98.3 | 96.4 | 98.7 | 99.8 | 87.8 | 97.3 | 97.3 | 98.3 | 88.5 | **84.3** |
| D$^3$-MoE (Conservative) | 98.0 | 94.7 | 98.3 | 99.7 | 86.5 | 96.7 | 95.8 | 98.2 | 88.4 | 81.5 |
| D$^3$-MoE (Aggressive) | 96.1 | 91.2 | 96.8 | 99.5 | 91.3 | 93.4 | 95.2 | 97.9 | 86.3 | 76.7 |
| **D$^3$-MoE (Best-of-Three)** | 98.5 | 97.6 | 99.0 | 99.8 | 91.0 | 97.6 | 98.1 | 98.3 | 91.3 | **87.5** |

结论：Aggressive 分支 EP 最高（91.3），但安全指标下降；Conservative 分支更谨慎但 progress 受损；Best-of-Three 能在场景级别选择更合适的风格。

### 6.5 Best-of-Three 归因分析

论文统计 12,147 个 aligned validation samples，分析哪个风格贡献了最终最优解。

| Style Expert | Unique Optimum | Tied Optimum | Effective Contribution |
|--------------|----------------|--------------|------------------------|
| Aggressive | 6,236 | 22 | 6,258 (51.52%) |
| Normal | 1,438 | 4,046 | 5,484 (45.15%) |
| Conservative | 383 | 22 | 405 (3.33%) |

这组结果很关键：Aggressive 单独评估较低，却在 Best-of-Three 里贡献最多最优解，说明多风格候选不是简单“平均性能更好”，而是在不同场景中提供不同上界。

### 6.6 风格运动学分析与定性结果

> **图 4：Trajectory features across three styles on the navtest benchmark.**（对应论文 Figure 4）
>
> ![D3-MoE 三风格运动学分布](./imgs/d3moe_fig4_style_violin.png)
>
> - **Aggressive**：Mean Speed、Mean Abs. Accel.、Mean Abs. Jerk 的均值和方差更高，说明它更倾向主动加速。
> - **Conservative**：纵向动力学更克制，但在弯道上可能因为严格 lane-centering 出现更高累计 yaw 与 lateral acceleration。
> - **Normal**：整体处于两者之间，是 balanced baseline。

> **图 5：Visualization of style-controllable trajectories. Each urban scenario pairs a BEV map (left) with a corresponding front-view image (right).**（对应论文 Figure 5）
>
> ![D3-MoE 风格可控轨迹可视化](./imgs/d3moe_fig5_qualitative.jpg)
>
> - **Aggressive** 轨迹更偏向大转弯半径、更高速度和更强 progress。
> - **Conservative** 轨迹更收敛、速度更低，更强调安全边界。
> - **Normal** 在效率和安全之间折中。

> **图 6：Visualization of all longitudinal-lateral expert pairings at an intersection.**（对应论文 Figure 6）
>
> ![D3-MoE 专家组合可解释性](./imgs/d3moe_fig6_prompt_control.png)
>
> - 图中枚举 3 个纵向专家与 5 个横向专家的 15 种组合。
> - 红框表示动态 router 在正常推理中选择的组合（Braking + Sharp Right），与 GT 转向语义一致。
> - 这说明 D$^3$-MoE 的专家不只是提高容量，也形成了可解释、可组合的行为空间。

### 6.7 消融实验：核心模块

| ID | MoE | Decoupling | Route Supervision | NC | DAC | EP | TTC | C | PDMS |
|----|-----|------------|-------------------|----|-----|----|-----|---|------|
| 1 | yes | yes | yes | 98.3 | 96.4 | 82.7 | 94.3 | 99.9 | 88.2 |
| 2 | yes | yes | no | 98.1 | 95.3 | 81.4 | 93.8 | 100 | 86.9 |
| 3 | yes | no | yes | 98.1 | 95.4 | 81.5 | 94.1 | 99.9 | 87.0 |
| 4 | no | yes | N/A | 98.2 | 95.8 | 82.2 | 94.3 | 99.9 | 87.8 |
| 5 | no | no | N/A | 98.0 | 96.2 | 82.4 | 93.7 | 99.9 | 87.6 |

结论：

1. 只有横纵解耦、不用 MoE，PDMS 从 87.6 到 87.8，收益较小。
2. 加 MoE 但无 routing supervision，会降到 86.9，说明专家容量本身不够，router 稳定性是关键。
3. 完整模型 88.2，证明解耦 action space + supervised routing 的组合最有效。

### 6.8 消融实验：Diffusion Denoising Steps

| Denoising Steps | NC | DAC | EP | TTC | C | PDMS | Inference Time (ms) |
|-----------------|----|-----|----|-----|---|------|---------------------|
| 1 | 98.1 | 96.2 | 82.4 | 94.2 | 99.9 | 88.0 | 41.14 |
| 2 | 98.3 | 96.4 | 82.7 | 94.3 | 99.9 | 88.2 | 58.83 |
| 3 | 98.3 | 96.4 | 82.7 | 94.3 | 99.9 | 88.2 | 69.94 |

论文默认采用 2 steps，因为 3 steps 不提升 PDMS，只增加延迟。

### 6.9 消融实验：固定专家组合

禁用动态 routing 后，穷举 15 种固定组合，PDMS 最高也只有 76.09，远低于动态 routing 的 88.2。

| Longitudinal Expert | Lat-E1 | Lat-E2 | Lat-E3 | Lat-E4 | Lat-E5 |
|---------------------|--------|--------|--------|--------|--------|
| Lon-E1 | 35.22 | 64.31 | 61.38 | **76.09** | 38.38 |
| Lon-E2 | 30.95 | 57.06 | 53.34 | 70.48 | 35.27 |
| Lon-E3 | 22.19 | 44.87 | 41.24 | 59.16 | 27.34 |

结论：专家需要按场景动态组合；把某个专家组合固定为全局策略会严重崩溃。

---

## 7. KnowHow（核心洞察）

1. **style-averaging 不是小 bug，而是回归目标的统计后果**：当相似输入对应多种有效动作时，单点回归会天然趋向条件均值。
2. **多模态生成必须配选择机制**：扩散模型能采样多模态，但自动驾驶最终要执行一条轨迹，因此 D$^3$-MoE 把 generation 和 selection 明确拆开。
3. **MoE 的关键不是专家数量，而是 routing 语义**：无 supervision 的 MoE 反而掉点，说明专家如果不能对应稳定物理行为，只会引入路由噪声。
4. **横纵解耦是自动驾驶规划里的强 inductive bias**：横向负责转向/航向，纵向负责速度/加速度，二者既独立又通过 lateral-fusion 保持一致。
5. **用 GT 运动学做自监督标签比人工场景标签更开放**：场景标签离散且主观，运动学特征连续且直接对应轨迹物理属性。
6. **Aggressive 分支不是为了单独部署高分，而是扩展候选上界**：它单独 PDMS 低，但在 Best-of-Three 中贡献 51.52% 的最优样本。
7. **2-step diffusion 是规划任务的实用折中**：NAVSIM 上 2 steps 已达峰值，更多 denoising step 不一定带来收益。
8. **可控性需要结构化接口**：style-conditioned AdaLN、显式 style index/language 入口、可解释专家组合，共同构成用户可控规划接口。

---

## 8. arXiv Appendix 关键点总结

论文 arXiv v1 源码未提供独立 Appendix，也没有 A/B/C/D/E/F/G 附录结构。以下按正文中可复核的补充信息整理：

| 项目 | 正文位置 | 关键点 |
|------|----------|--------|
| 数据设置 | Experiments / Dataset Setup | NAVSIM；8 cameras、5 LiDAR sensors、map annotations、3D boxes；2 秒历史、4 秒未来、8 waypoints |
| 多风格增强 | Dataset and Stylized Reference Trajectory Augmentation | Normal=GT；Aggressive=1.05-1.25x 速度增强；Conservative=减速、增大跟车距离、车道居中与 horizon truncation |
| 超参数 | Table I | 横向 5 专家、纵向 3 专家、Top-1 激活、100 epochs、batch 32、LR 1e-4、2 denoising steps |
| 指标定义 | Evaluation Metrics | PDMS 包含 NC/TTC/DAC/C/EP；EPDMS 额外包含 LK/HC/EC，并对 DDC/TLC 严重违规做乘性惩罚 |
| 动态路由有效性 | Fixed Expert Combination Table | 固定任一专家组合都会大幅掉点，最高 76.09 PDMS，支持动态 routing 必要性 |
| 风格互补性 | Best-of-Three Attribution | Aggressive、Normal、Conservative 分别贡献 51.52%、45.15%、3.33% 最优样本 |

---

## 9. 总结

D$^3$-MoE 的核心价值在于把端到端自动驾驶规划里的两个混在一起的问题拆开：**一是行为风格的多模态，二是轨迹物理分量的可解释控制**。行为上，它不再强迫模型输出单一平均风格，而是并行生成 aggressive / normal / conservative 候选；物理上，它让 lateral 与 longitudinal 专家各自学习清晰语义，并通过自监督 routing 按场景激活。

三大贡献可以概括为：

1. **风格可控扩散规划**：用 style-conditioned diffusion 逃离 style-averaging，提供用户偏好和 evaluator 选择接口。
2. **横纵解耦 MoE**：用 5 个 lateral experts 与 3 个 longitudinal experts 形成可解释专家组合，并通过 lateral-fusion 保持运动学一致。
3. **NAVSIM 强结果**：Normal 达到 88.2 PDMS / 84.3 EPDMS，Best-of-Three 达到 91.3 PDMS / 87.5 EPDMS。

最重要洞察是：**自动驾驶里的多模态不是“随机采样更多轨迹”就能解决，必须把候选空间、选择机制、物理语义和用户偏好一起设计**。D$^3$-MoE 的结构化解耦正是围绕这一点展开。

---

## 参考链接

| 资源 | 链接 |
|------|------|
| **论文** | [arXiv:2606.04884](https://arxiv.org/abs/2606.04884) |
| **PDF** | [arXiv PDF](https://arxiv.org/pdf/2606.04884) |
| **HTML** | [arXiv HTML](https://arxiv.org/html/2606.04884v1) |
| **代码** | [待验证] arXiv 页面未列出官方代码链接 |
| **项目主页** | [待验证] arXiv 页面未列出官方项目主页 |
