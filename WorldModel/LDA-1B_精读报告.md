# LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion

## 引用信息

| 字段 | 内容 |
|------|------|
| **标题** | LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion |
| **arXiv** | [2602.12215](https://arxiv.org/abs/2602.12215) [cs.RO] |
| **作者** | Jiangran Lyu*, Kai Liu*, Xuheng Zhang*, Haoran Liao, Yusen Feng, Wenxuan Zhu, Tingrui Shen, Jiayi Chen, Jiazhao Zhang, Yifei Dong, Wenbo Cui, Senmao Qi, Shuo Wang, Yixin Zheng, Mi Yan, Xuesong Shi, Haoran Li, Dongbin Zhao, Ming-Yu Liu, Zhizheng Zhang†, Li Yi†, Yizhou Wang†, He Wang† |
| **机构** | 北京大学 · Galbot · 中科院自动化所 · 北京智源 (BAAI) · 清华大学 · 中山大学 · NVIDIA |
| **顶会/顶刊** | RSS 2026 |
| **发布日期** | 2026-02-12 |

---

## 1. Motivation（问题背景）

### 1.1 机器人基础模型的核心困境

受大语言模型（LLM）和视觉语言模型（VLM）成功的启发，机器人社区越来越追求通用机器人基础模型。然而，现有方法大多围绕**行为克隆（Behavior Cloning）** 范式，仅从高质量专家演示中模仿动作，根本性地将学习限制在高质量演示数据上。

### 1.2 异构具身数据的浪费

大量异构具身数据——低质量轨迹、人类操作视频、无标注交互过程——蕴含着极其宝贵的**物理先验**和**交互动力学**知识，却因"动作最优性"这一单一标准被直接丢弃。现有方法把"动作最优性"当成了唯一货币，却忽视了"世界如何响应动作"这一更可迁移的知识。

### 1.3 统一世界模型（UWM）的局限

| 局限 | 说明 |
|------|------|
| **粗放的数据使用** | 异构数据被统一对待，不区分质量和监督角色，浪费可迁移的动力学知识 |
| **缺乏统一大规模数据集** | 现有具身数据集高度碎片化，格式、传感器、动作表示、标注质量各异 |
| **像素空间表征** | 在像素空间预测未来状态，将动力学学习与冗余外观建模纠缠，训练效率低 |

### 1.4 三大核心问题

1. **数据利用不充分**：BC 范式只能高效吸收高质量演示，低质量数据和无动作视频被浪费
2. **表征空间不合适**：像素空间/VAE 空间纠缠外观与动力学，语义结构不足，阻碍规模化
3. **数据集碎片化**：缺乏统一格式、对齐动作表示的大规模具身交互数据集

### 1.5 LDA-1B 的核心出发点

北京大学与 Galbot 等团队提出：**机器人基础模型想真正扩展到 foundation-level，就不能只学"答案"（动作），还必须学"因果过程"（动力学）**。通过**通用具身数据摄取（Universal Embodied Data Ingestion）** 让不同质量数据各尽其用，在**结构化 DINO 潜在空间**中联合学习策略、动力学和视觉预测，实现 1B 参数规模的稳定训练。

---

## 2. 一句话总结

LDA-1B 通过**通用具身数据摄取**统一利用 30k+ 小时异构数据（高质量动作 / 低质量轨迹 / 无动作视频），在**结构化 DINO 潜在空间**中联合学习策略、前向/逆向动力学与视觉预测，以 1B 参数在接触丰富、灵巧操作和长时程任务上分别超越 π₀.₅ 达 21%、48% 和 23%。

---

## 拟人化开篇

想象你在教一个机器人做家务。你有三种"教材"：专业厨师的完美操作录像（高质量+有动作标注）、新手笨拙但还算完整的练习录像（低质量+有动作标注）、以及大量网上随手拍的烹饪视频（无动作标注）。

当前大多数机器人模型只会用第一种教材——因为它们只会"照葫芦画瓢"地模仿专家动作。第二种教材动作不够标准，怕学坏了；第三种连动作标注都没有，更没法用。

但仔细想想：新手虽然笨拙，但他碰倒杯子后如何恢复、锅铲打滑后怎么调整——这些**动力学知识**恰恰是专家演示中看不到的！而那些随手拍的视频，虽然没标注动作，却展示了物体如何被推动、如何倾倒——这正是**世界如何响应动作**的宝贵知识。

LDA-1B 的核心思路就是：**让每种数据各尽其用**——专家数据学"该做什么"，低质量数据学"世界如何响应"，无标注视频学"未来会怎样变化"。三者合一，才是真正的机器人基础模型。

---

## 背景与问题动机

### 机器人基础模型的演进

| 范式 | 代表工作 | 数据需求 | 局限 |
|------|---------|---------|------|
| **行为克隆 (BC)** | π₀、RDT、InternVLA | 高质量专家演示 | 只能利用高质量数据，规模化受限 |
| **动作对齐 + BC** | Being-H0、UniVLA | 混合质量 | 依赖预训练潜在动作模型，有效数据规模约 6k 小时 |
| **统一世界模型 (UWM)** | UWM、UVA、Motus | 混合质量 | 像素空间预测，不区分数据角色，规模化困难 |
| **LDA-1B** | 本文 | 30k+ 小时异构数据 | DINO 潜在空间 + 角色感知数据使用 |

![图 1：LDA-1B 核心思想总览](https://arxiv.org/html/2602.12215v1/x1.png)

> **图 1**：LDA-1B 不是单纯扩大参数规模，而是通过统一摄取 30k+ 小时异构具身数据，将高质量动作数据、低质量动作数据与无动作视频分别分配到策略学习、动力学学习和视觉预测中，在 DINO 潜在空间里联合建模。

### 统一世界模型（UWM）框架

UWM 给出了联合建模的统一框架，同时学习四个条件分布：

1. **策略 (Policy)**：$p(\boldsymbol{a}_{t+1:t+k} \mid \boldsymbol{o}_t)$ ——给定当前观测，预测未来动作
2. **前向动力学 (Forward Dynamics)**：$p(\boldsymbol{o}_{t+1:t+k} \mid \boldsymbol{o}_t, \boldsymbol{a}_{t+1:t+k})$ ——给定观测和动作，预测未来状态
3. **逆向动力学 (Inverse Dynamics)**：$p(\boldsymbol{a}_{t+1:t+k} \mid \boldsymbol{o}_{t:t+k})$ ——给定状态转移，推断动作
4. **视觉规划 (Visual Planning)**：$p(\boldsymbol{o}_{t+1:t+k} \mid \boldsymbol{o}_t)$ ——给定当前观测，预测未来视觉状态

但现有 UWM 实现在像素空间操作，且不区分数据角色，限制了规模化。

---

## 方法详解

### 3.1 整体架构

LDA-1B 由四个核心设计组成：通用数据摄取、DINO 潜在表征、MM-DiT 架构、预训练+后训练流程。

![图 2：LDA 架构图](https://arxiv.org/html/2602.12215v1/x2.png)

> **图 2**：LDA 架构图。模型在多个共训练目标下联合去噪动作块和未来视觉潜在特征，包括策略学习、前向动力学、逆向动力学和视觉预测。以 VLM tokens、扩散时间步和任务嵌入为条件，采用多模态扩散 Transformer 架构，动作和视觉专家解耦但通过共享自注意力层交互。

### 3.2 Universal Data Ingestion：通用数据摄取

#### 3.2.1 核心思想

异构数据按监督质量分配不同角色：

| 数据类型 | 质量 | 可用目标 | 角色 |
|---------|------|---------|------|
| **高质量机器人/人类演示** | 高 | 策略 + 动力学 + 视觉预测 | 全部目标 |
| **低质量轨迹**（含次优动作） | 低 | 动力学 + 视觉预测 | 动力学学习不需要动作最优性 |
| **无动作人类视频** | 无动作标注 | 视觉预测 | 指令条件下的未来状态预测 |

这种**角色感知的数据使用**防止了对专家行为的过拟合，实现了可迁移动力学和动作表征的可扩展学习。

#### 3.2.2 任务嵌入与寄存器 Token

为实现单一扩散模型内的差异化目标，引入：

- **4 个可学习任务嵌入**：分别对应策略、前向动力学、逆向动力学、视觉预测，加到扩散时间步嵌入 $f_t$ 上
- **2 个可学习寄存器 Token**：一个动作寄存器、一个视觉寄存器，作为缺失模态的占位符

例如：策略训练时，模型接收带噪动作 token + 视觉寄存器 token（代表未观测的未来状态）；视觉预测时，接收带噪未来视觉 token + 动作寄存器 token。

##### 伪代码：Task Embedding 生成与控制

> 源码参考：MMDiT_ActionHeader.py

```python
# === 1. Task Embedding 定义（nn.Parameter，非 nn.Embedding）===
# 维度 = num_attention_heads × attention_head_dim（如 DiT-L: 32×48=1536）
self.policy_embedding = nn.Parameter(0.02 * torch.randn(inner_dim))   # 策略
self.fd_embedding     = nn.Parameter(0.02 * torch.randn(inner_dim))   # 前向动力学
self.vg_embedding     = nn.Parameter(0.02 * torch.randn(inner_dim))   # 视觉预测
self.id_embedding     = nn.Parameter(0.02 * torch.randn(inner_dim))   # 逆向动力学

# === 2. 缺失模态的可学习替代 Token ===
# 动作缺失时（视觉预测/前向动力学任务），用 action_learnable_tokens 替代
self.action_learnable_tokens = nn.Embedding(action_horizon, input_embedding_dim)
# 视觉缺失时（策略/逆向动力学任务），用 next_obs_learnable_tokens 替代
self.next_obs_learnable_tokens = nn.Parameter(0.02 * torch.randn(num_chans))

# === 3. Register Tokens（附加到视觉 token 前的额外"暂存空间"）===
self.register_tokens = nn.Embedding(num_target_vision_tokens, input_embedding_dim)

# === 4. 训练时：按任务分组，为每组分配对应 Task Embedding ===
# 根据每个样本的 assigned_task 分组
policy_indices = [i for i, t in enumerate(tasks) if t == "policy"]
fd_indices     = [i for i, t in enumerate(tasks) if t == "forward_dynamics"]
vg_indices     = [i for i, t in enumerate(tasks) if t == "video_gen"]
id_indices     = [i for i, t in enumerate(tasks) if t == "inverse_dynamics"]

# 扩展到对应样本数
policy_emb = self.policy_embedding.unsqueeze(0).expand(len(policy_indices), -1)
fd_emb     = self.fd_embedding.unsqueeze(0).expand(len(fd_indices), -1)
vg_emb     = self.vg_embedding.unsqueeze(0).expand(len(vg_indices), -1)
id_emb     = self.id_embedding.unsqueeze(0).expand(len(id_indices), -1)

# 按任务顺序拼接
task_embedding = torch.cat((policy_emb, id_emb, fd_emb, vg_emb), dim=0)  # (B, inner_dim)

# === 5. Task Embedding 注入方式：加到 Timestep Embedding 上 ===
# 在 MMDiT 模型内部：
time_cond = self.timestep_encoder(diffusion_t)   # (B, inner_dim) 正弦时间步编码
if task_embedding is not None:
    time_cond += task_embedding                    # 简单相加！
# → time_cond 随后通过 AdaLN 控制所有 Transformer 块

# === 6. 各任务的模态输入构造 ===
# | 任务           | 动作输入              | 视觉输入              | 加噪目标  | 预测目标       |
# |---------------|----------------------|----------------------|----------|---------------|
# | policy        | 带噪动作 (flow t)     | learnable_obs_tokens | 动作     | 动作 velocity  |
# | forward_dyn   | 干净动作 (t=1)        | 带噪视觉 (flow t)     | 视觉     | 视觉 velocity  |
# | inverse_dyn   | 带噪动作 (flow t)     | GT 视觉（不加噪）      | 动作     | 动作 velocity  |
# | video_gen     | learnable_act_tokens  | 带噪视觉 (flow t)     | 视觉     | 视觉 velocity  |

# === 7. 推理时：仅使用对应任务的 Task Embedding ===
# 策略推理：task_embedding = self.policy_embedding.unsqueeze(0).expand(B, -1)
# 视觉预测推理：task_embedding = self.vg_embedding.unsqueeze(0).expand(B, -1)
```

#### 3.2.3 训练目标

$$
\mathcal{L}^\theta_{\text{action}} = \mathbb{E}_{(\boldsymbol{o}_{t:t+k}, \boldsymbol{a}_{t+1:t+k}, \ell) \sim \mathcal{D}} \mathbb{E}_{\tau_a \sim \mathcal{U}(0, T_\tau)} \mathbb{E}_{\epsilon_a \sim \mathcal{N}(0, I)} \left\| v^\theta_a - (\epsilon_a - \boldsymbol{a}_{t+1:t+k}) \right\|_2^2
$$

$$
\mathcal{L}^\theta_{\text{obs}} = \mathbb{E}_{(\boldsymbol{o}_{t:t+k}, \boldsymbol{a}_{t+1:t+k}, \ell) \sim \mathcal{D}} \mathbb{E}_{\tau_o \sim \mathcal{U}(0, T_\tau)} \mathbb{E}_{\epsilon_o \sim \mathcal{N}(0, I)} \left\| v^\theta_o - (\epsilon_o - \boldsymbol{o}_{t+1:t+k}) \right\|_2^2
$$

$$
\mathcal{L}^\theta = \mathcal{L}^\theta_{\text{action}} + \mathcal{L}^\theta_{\text{obs}}
$$

训练时根据任务规范选择性激活动作和视觉损失；推理时通过指定任务嵌入和对应输入灵活调用不同目标。

##### 伪代码：Loss 计算详细设计

> 源码参考： MMDiT_ActionHeader.py

```python
# ============================================================
# Loss 计算：Flow Matching MSE Loss
# ============================================================
# 模型输出 pred_actions 和 pred_next_obs
# pred_actions: 仅取最后 action_horizon 个 token（排除 state/history 前缀）
pred_actions = action_tokens[:, -actions.shape[1]:]  # (B, T, action_dim)

# 视觉预测：obs_projector 将 hidden_size 投影回 DINO 通道维度
# 仅取最后 obs_len + glob_len 个 token（obs 部分，排除 register tokens）
pred_next_obs = self.obs_projector(image_tokens)
pred_next_obs = pred_next_obs[-len(obs_indices):, -(obs_len + glob_len):]

total_loss = 0.0

# ---- 1. Policy Loss（始终计算）----
# 目标：action_velocity = clean_action - noise（flow matching velocity）
# 掩码：action_mask 处理变长动作序列
policy_pred = pred_actions[:len(policy_indices)]
policy_target = action_velocity[:len(policy_indices)]
policy_loss = F.mse_loss(policy_pred, policy_target, reduction="none")
policy_loss = (policy_loss * action_mask[policy_indices]).sum() / (
    action_mask[policy_indices].sum() + 1e-8
)
total_loss += policy_loss

# ---- 2. Inverse Dynamics Loss（非 only_policy 时计算）----
# 仅监督前 future_obs_index 个时间步（对应未来观测的动作区间）
inverse_pred = pred_actions[len(policy_indices):len(pred_action_task_indices),
                            :future_obs_index]
inverse_target = action_velocity[len(policy_indices):, :future_obs_index]
inverse_loss = F.mse_loss(inverse_pred, inverse_target, reduction="none")
inverse_loss = (inverse_loss * action_mask[inverse_dynamics_indices,
                            :future_obs_index]).sum() / (
    action_mask[inverse_dynamics_indices, :future_obs_index].sum() + 1e-8
)
total_loss += inverse_loss

# ---- 3. Observation Loss（forward_dynamics + video_gen）----
# 无掩码，直接 MSE
if policy_and_video_gen:
    # 简化模式：仅 video_gen 的视觉损失
    pred_obs = pred_next_obs[-len(video_gen_indices):]
    obs_target = obs_velocity[-len(video_gen_indices):]
    obs_loss = F.mse_loss(pred_obs, obs_target)
elif only_wo_video_gen:
    # 仅 forward_dynamics 的视觉损失
    pred_obs = pred_next_obs[:len(forward_dynamics_indices)]
    obs_target = obs_velocity[:len(forward_dynamics_indices)]
    obs_loss = F.mse_loss(pred_obs, obs_target)
else:
    # 完整模式：所有视觉预测任务的损失
    pred_obs = pred_next_obs
    obs_target = obs_velocity
    obs_loss = F.mse_loss(pred_obs, obs_target)
total_loss += obs_loss

# ---- 返回 ----
return {
    "loss": total_loss,
    "action_loss": policy_loss,      # 策略损失
    "dynamics_loss": obs_loss,        # 动力学/视觉预测损失
}

# ============================================================
# 训练模式选择（由配置标志控制）
# ============================================================
# | 模式                     | 训练任务                              | 回退策略                        |
# |--------------------------|--------------------------------------|-------------------------------|
# | only_policy=True         | policy only                          | —                              |
# | policy_and_video_gen=True| policy + video_gen                   | 未匹配→policy(偶)/video_gen(奇) |
# | only_wo_video_gen=True   | policy + forward_dyn + inverse_dyn   | 未匹配→policy                   |
# | default（全部4任务）      | policy + fd + id + vg                | 未匹配→policy                   |
```

### 3.3 预测目标的表征

#### 3.3.1 视觉预测：DINO 潜在特征

采用预训练 DINO 编码器提取的潜在特征，而非 VAE 像素空间表征。

**关键洞察**：DINO 潜在特征编码高层语义和空间结构，同时抑制背景噪声和低层视觉变化，有利于学习跨环境、跨物体配置泛化的场景动力学。

#### 3.3.2 动作表征：统一手中心动作空间

基于末端执行器运动的统一手中心动作空间：

- **平行夹爪**：6-DoF 手腕位姿增量 + 单自由度夹爪宽度
- **多指灵巧手**：6-DoF 手腕位姿增量 + 手腕坐标系下的关键点描述

#### 3.3.3 异步时间流

视觉状态和动作组织为两个同步时间流，**采样率不同**：

- **视觉观测**：3 Hz（低频，减少冗余计算）
- **动作**：10 Hz（高频，保留精细动作动态）

这保持了快速变化控制信号与慢速演化视觉状态之间的时间对齐。

### 3.4 MM-DiT：多模态扩散 Transformer

#### 3.4.1 整体设计

采用 MM-DiT 在统一扩散框架内联合去噪动作块和预测未来视觉特征。

**条件输入**：
- 当前观测 + 语言指令 → 预训练 VLM 编码为条件 token
- 扩散时间步 → 正弦嵌入
- 任务信息 → 可学习任务嵌入

所有条件信号通过 **AdaLN** 注入每个 Transformer 块。

#### 3.4.2 模态专家 + 共享注意力

- 动作和视觉各有**模态特定 QKV 投影和 FFN**（保留归纳偏置）
- **自注意力跨模态共享**（实现跨模态交互）
- 语言 token 通过**交叉注意力**提供高层语义指导
- 模态特定输出头预测去噪后的动作序列和未来视觉特征

##### 伪代码：MM-DiT Block 详细设计

> 源码参考：mmdit_cross_attn.py

```python
# ============================================================
# MM-DiT Block（Cross-Attention 变体，实际使用的版本）
# ============================================================
# 每个 Block 有 6 个残差流：image×2 + action×2 + 各自 FFN×2
# self.image_attn_residual_fn      # image self-attn 残差
# self.image_cross_attn_residual_fn # image cross-attn 残差
# self.image_ff_residual_fn         # image FFN 残差
# self.action_attn_residual_fn      # action self-attn 残差
# self.action_cross_attn_residual_fn # action cross-attn 残差
# self.action_ff_residual_fn        # action FFN 残差

class MMDiTBlock:
    def __init__(self, dim, num_heads, cross_attention_dim):
        # === AdaLN 条件投影：time_cond → 12 个调制参数 ===
        # 8 个 gamma（缩放）+ 4 个 beta（偏移）
        # image: pre_attn_gamma, post_attn_gamma, pre_ff_gamma, post_ff_gamma
        # action: pre_attn_gamma, post_attn_gamma, pre_ff_gamma, post_ff_gamma
        # image: pre_attn_beta, pre_ff_beta
        # action: pre_attn_beta, pre_ff_beta
        # 注意：post_attn 和 post_ff 只有 gamma（缩放），没有 beta（偏移）
        self.to_cond = nn.Sequential(
            Rearrange('b d -> b 1 d'),
            nn.SiLU(),
            nn.Linear(dim, 8 * dim + 4 * dim)  # 8 gammas + 4 betas
        )
        # 关键初始化：gamma → 1, beta → 0（零初始化保证训练初期为恒等映射）
        nn.init.zeros_(linear.weight)
        nn.init.zeros_(linear.bias)
        nn.init.constant_(linear.bias[:8*dim], 1.)  # gammas 初始化为 1

        # === 联合自注意力（image + action 共享）===
        self.joint_attn = JointAttention(dim, num_heads, dim_inputs=(dim, dim))

        # === 交叉注意力（各自独立 attend to text）===
        self.img_cross_attn = CrossAttention(dim, cross_attention_dim, num_heads)
        self.action_cross_attn = CrossAttention(dim, cross_attention_dim, num_heads)

        # === 模态特定 FFN ===
        self.image_ff = FeedForward(dim)
        self.action_ff = FeedForward(dim)

    def forward(self, image_tokens, action_tokens, text_tokens, time_cond, text_mask):
        # time_cond = timestep_embedding + task_embedding, shape: (B, dim)

        # ---- Step 1: AdaLN 调制参数生成 ----
        cond = self.to_cond(time_cond)  # (B, 1, 12*dim)
        # 拆分为 12 个参数，每个 (B, 1, dim)
        img_pre_attn_g, img_post_attn_g, img_pre_ff_g, img_post_ff_g,  # image gammas
        act_pre_attn_g, act_post_attn_g, act_pre_ff_g, act_post_ff_g,  # action gammas
        img_pre_attn_b, img_pre_ff_b,                                  # image betas
        act_pre_attn_b, act_pre_ff_b = split(cond)                     # action betas

        # ---- Step 2: 联合自注意力（image ↔ action 交互）----
        # AdaLN pre-attention：gamma * x + beta
        image_tokens = image_tokens * img_pre_attn_g + img_pre_attn_b
        action_tokens = action_tokens * act_pre_attn_g + act_pre_attn_b

        # 联合自注意力：image 和 action 互相 attend
        image_tokens, action_tokens = self.joint_attn(
            inputs=(image_tokens, action_tokens)
        )

        # AdaLN post-attention：仅 gamma 缩放（无偏移）
        image_tokens = image_tokens * img_post_attn_g
        action_tokens = action_tokens * act_post_attn_g

        # ---- Step 3: 交叉注意力（各自 attend to text）----
        image_tokens = image_tokens + self.img_cross_attn(
            image_tokens, encoder_hidden_states=text_tokens, attention_mask=text_mask
        )
        action_tokens = action_tokens + self.action_cross_attn(
            action_tokens, encoder_hidden_states=text_tokens, attention_mask=text_mask
        )

        # ---- Step 4: FFN + AdaLN ----
        # AdaLN pre-FF
        image_tokens = image_tokens * img_pre_ff_g + img_pre_ff_b
        action_tokens = action_tokens * act_pre_ff_g + act_pre_ff_b

        # FFN
        image_tokens = image_tokens + self.image_ff(image_tokens)
        action_tokens = action_tokens + self.action_ff(action_tokens)

        # AdaLN post-FF：仅 gamma 缩放
        image_tokens = image_tokens * img_post_ff_g
        action_tokens = action_tokens * act_post_ff_g

        return image_tokens, action_tokens
```

##### 伪代码：MM-DiT 顶层前向流程

```python
# ============================================================
# MM-DiT 顶层（MMDiT class）
# ============================================================
class MMDiT:
    def __init__(self, inner_dim, num_blocks, cross_attention_dim, output_dim):
        self.timestep_encoder = TimestepEncoder(inner_dim)
        self.text_attn_layernorm = nn.LayerNorm(cross_attention_dim, elementwise_affine=False)
        self.blocks = nn.ModuleList([MMDiTBlock(inner_dim, ...) for _ in range(num_blocks)])
        self.action_proj_out = nn.Linear(inner_dim, output_dim)
        self.image_proj_out = nn.Linear(inner_dim, output_dim)

    def forward(self, image_tokens, action_tokens, text_tokens,
                register_tokens, text_mask, time_cond, task_embedding):
        # ---- 1. Register Tokens 前置到 image tokens ----
        if register_tokens is not None:
            # register_tokens: (B, num_registers, dim)
            image_tokens, packed_shape = pack([register_tokens, image_tokens], 'b * d')

        # ---- 2. 扩展残差流（hyper connections）----
        image_tokens = self.expand_streams(image_tokens)
        action_tokens = self.expand_streams(action_tokens)

        # ---- 3. 文本 tokens 归一化（仅一次，不逐块归一化）----
        text_tokens = self.text_attn_layernorm(text_tokens)

        # ---- 4. 编码时间步 + 注入 Task Embedding ----
        time_cond = self.timestep_encoder(time_cond)  # (B, inner_dim)
        if task_embedding is not None:
            time_cond += task_embedding                 # 简单相加

        # ---- 5. 逐块处理 ----
        for block in self.blocks:
            image_tokens, action_tokens = block(
                image_tokens, action_tokens, text_tokens,
                time_cond=time_cond, text_mask=text_mask
            )

        # ---- 6. 移除 Register Tokens ----
        if register_tokens is not None:
            _, image_tokens = unpack(image_tokens, packed_shape, 'b * d')

        # ---- 7. 归一化 + 投影输出 ----
        image_tokens = self.norm(image_tokens)
        action_tokens = self.action_norm(action_tokens)
        action_out = self.action_proj_out(action_tokens)
        image_out = self.image_proj_out(image_tokens)

        return image_out, action_out
```

##### 伪代码：时间步采样与加噪流程

> 源码参考：[MMDiT_ActionHeader.py](file:///E:/github/code/LDA-1B-main/lda/model/modules/action_model/MMDiT_ActionHeader.py), [action_encoder.py](file:///E:/github/code/LDA-1B-main/lda/model/modules/action_model/flow_matching_head/action_encoder.py)

```python
# ============================================================
# 时间步采样：Beta 分布偏置向 t≈1（干净数据）
# ============================================================
self.beta_dist = Beta(alpha=1.5, beta=1.0)  # 偏置向大值

def sample_time(self, batch_size, device, dtype):
    sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
    return (0.999 - sample) / 0.999  # 映射到约 [0, 1]

# ============================================================
# 动作加噪（policy + inverse_dynamics 任务）
# ============================================================
# 仅对需要预测动作的任务加噪
to_noise_action = torch.cat((policy_action, inverse_action), dim=0)

# 独立采样动作时间步
act_t_sample = self.sample_time(B_action, device, dtype).reshape(-1)
act_t = act_t_sample[:, None, None]  # (B, 1, 1) 广播

# Flow matching 线性插值：noisy = (1-t)*noise + t*clean
action_noise = torch.randn_like(to_noise_action)
noisy_action = (1 - act_t) * action_noise + act_t * to_noise_action

# Flow matching velocity 目标：v = clean - noise
action_velocity = to_noise_action - action_noise

# 离散化时间步（用于 TimestepEncoder）
act_t_discretized = (act_t_sample[:, 0, 0] * num_timestep_buckets).long()

# ============================================================
# 视觉加噪（forward_dynamics + video_gen 任务）
# ============================================================
to_noise_next_obs = torch.cat((forward_obs, video_gen_obs), dim=0)

# 独立采样视觉时间步（与动作时间步独立！）
obs_t_sample = self.sample_time(B_obs, device, dtype).reshape(-1)
obs_t = obs_t_sample[:, None, None]

obs_noise = torch.randn_like(to_noise_next_obs)
noisy_obs = (1 - obs_t) * obs_noise + obs_t * to_noise_next_obs
obs_velocity = to_noise_next_obs - obs_noise

obs_t_discretized = (obs_t_sample[:, 0, 0] * num_timestep_buckets).long()

# ============================================================
# 前向动力学：动作以 t=1（干净）编码
# ============================================================
t_clean = torch.ones(len(forward_dynamics_indices), device=device)
t_discretized_clean = (t_clean * num_timestep_buckets).long()
forward_act_feat = self.action_encoder(actions[forward_dynamics_indices],
                                        t_discretized_clean, ...)

# ============================================================
# 历史动作：也以 t=1（干净）编码
# ============================================================
history_t_clean = torch.ones(history_actions.shape[0], device=device)
history_t_discretized = (history_t_clean * num_timestep_buckets).long()
history_action_features = self.action_encoder(history_actions,
                                               history_t_discretized, ...)

# ============================================================
# 合并扩散时间步（每个样本只有一个时间步）
# ============================================================
diffusion_t = torch.cat((act_t_discretized, obs_t_discretized), dim=0)  # (B,)

# ============================================================
# Action Encoder：将动作+时间步编码为 token
# ============================================================
class ActionEncoder:
    def __init__(self, action_dim, hidden_size):
        self.W1 = nn.Linear(action_dim, hidden_size)        # d → w
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)   # 2w → w
        self.W3 = nn.Linear(hidden_size, hidden_size)        # w → w
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        # actions: (B, T, action_dim), timesteps: (B,)
        timesteps = timesteps.unsqueeze(1).expand(-1, T)    # (B,) → (B, T)
        a_emb = self.W1(actions)                             # (B, T, w)
        tau_emb = self.pos_encoding(timesteps)               # (B, T, w)
        x = torch.cat([a_emb, tau_emb], dim=-1)             # (B, T, 2w)
        x = swish(self.W2(x))                                # (B, T, w)
        x = self.W3(x)                                       # (B, T, w)
        return x

# 多具身形态版本：W1/W2/W3 替换为 CategorySpecificLinear
# CategorySpecificLinear: 每个具身形态有独立的权重矩阵 W[cat_id] 和偏置 b[cat_id]
# forward: x @ W[cat_id] + b[cat_id]
```

##### 伪代码：推理时 Euler 积分

```python
# ============================================================
# 策略推理：Euler 积分去噪动作
# ============================================================
def policy_inference(self, obs_tokens, text_tokens, action_dim, num_steps=4):
    actions = torch.randn(B, action_horizon, action_dim)  # 从噪声开始
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t_cont = step / num_steps
        t_discretized = int(t_cont * num_timestep_buckets)

        # 编码当前带噪动作
        action_features = self.action_encoder(actions, t_discretized)

        # 构造输入（obs 用 learnable_next_obs_tokens 替代）
        obs_input = obs_tokens  # 当前观测的 DINO 特征
        next_obs_placeholder = self.next_obs_learnable_tokens  # 视觉占位符

        # MM-DiT 前向
        _, action_tokens = self.model(
            image_tokens=obs_input, action_tokens=action_features,
            text_tokens=text_tokens, time_cond=t_discretized,
            task_embedding=self.policy_embedding,  # 策略任务嵌入
            register_tokens=self.register_tokens.weight,
        )

        # 解码预测 velocity
        pred_velocity = self.action_decoder(action_tokens)[:, -action_horizon:]

        # Euler 步进
        actions = actions + dt * pred_velocity

    return actions  # 去噪后的动作
```

### 3.5 预训练与后训练

#### 预训练配置

| 配置 | 值 |
|------|-----|
| GPU | 48× NVIDIA H800 |
| 训练迭代 | 400K |
| 总计算量 | 4,608 GPU 小时 |
| 冻结模块 | VLM (Qwen3-VL) + DINO 编码器 |
| 可训练模块 | MM-DiT + 动作编码器/解码器 |

**关键设计**：保持 VLM 和 DINO 冻结，确保跨模态理解和细粒度视觉特征提取的核心能力不被破坏。

#### 数据高效微调

轻量级后训练阶段，遵循与预训练相同的数据机制，**直接利用未过滤的混合质量遥操作数据**，无需专家级演示。相比依赖精心筛选专家数据集的微调流程，大幅提高数据效率，降低数据收集和标注成本。

---

## EI-30K：具身交互数据集

### 4.1 数据集概览

![图 4：EI-30K 统计信息](https://arxiv.org/html/2602.12215v1/x4.png)

> **图 4**：EI-30K 数据集统计。包含 30k+ 小时多样化人类和机器人交互数据（右），跨越不同片段长度（左）和丰富的操作任务集（中）。

| 数据类型 | 来源/子数据集 | 时长（小时） |
|---------|-------------|------------|
| **真实世界机器人** | Open X-Embodiment | 3,000 |
| | Agibot World | 3,276 |
| | RoboMIND | 305 |
| | Humanoid Everyday | 30 |
| | RoboCOIN | 500 |
| | Galaxea | 500 |
| | LET | 1,000 |
| **仿真机器人** | InternData-A1 | 7,433 |
| | Behavior-1k | 1,200 |
| **自我中心人类（有动作）** | Ego4D | 3,670 |
| | Epic-Kitchens | 100 |
| | Ego-Exo4d | 1,286 |
| | SSV2 | 240 |
| | EgoDex | 830 |
| | HOT3D | 16 |
| | HoloAssist | 166 |
| | OAKINK2 | 6.5 |
| | TACO | 3.2 |
| | HOI4D | 7.6 |
| | ARCTIC | 2.3 |
| **自我中心人类（无动作）** | Egocentric-10k | 10,000 |
| | RH20T-human | 100 |
| | Egome | 80 |
| | Taste-Rob | 130 |
| **总计** | | **30k+** |

### 4.2 数据统一

- **格式统一**：所有数据转换为 LeRobot 格式，提供观测、动作、语言的统一表示
- **动作对齐**：所有动作标注表示为共享坐标系下的手中心运动

![图 3：对齐的末端执行器坐标系](https://arxiv.org/html/2602.12215v1/x3.png)

> **图 3**：跨多样机器人和人类具身形态手动对齐坐标系，确保一致性。共享表示使异构交互数据的联合学习成为可能。

- **质量标注**：每条轨迹分配质量标签（基于动作精度和标注完整性），低质量轨迹保留而非激进过滤

---

## 实验结果

### 5.1 仿真实验：RoboCasa-GR1

![图 5：真实世界操作演示](https://arxiv.org/html/2602.12215v1/x5.png)

> **图 5**：跨多机器人平台和末端执行器的真实世界操作演示。Galbot G1 配备 Sharpa 灵巧手（左上）、Unitree G1 配备 BrainCo 灵巧手（中下左）、Galbot G1 配备两指夹爪（右）。

| 模型 | 视觉表征 | MM-DiT | VLM | 成功率↑ |
|------|---------|--------|-----|---------|
| GR00T-N1.6 | - | - | Cosmos | 47.6 |
| StarVLA | - | - | Qwen3-VL | 47.8 |
| GR00T-EI30k | - | - | Qwen3-VL | 51.3 |
| UWM-0.1B | VAE | ✗ | ✗ | 14.2 |
| UWM-1B | VAE | ✗ | Qwen3-VL | 19.3 |
| UWM (MM-DiT) | VAE | ✓ | Qwen3-VL | 20.0 |
| LDA (DiT) | DINO | ✗ | Qwen3-VL | 48.9 |
| LDA-0.5B | DINO | ✓ | Qwen3-VL | 50.7 |
| **LDA-1B** | **DINO** | **✓** | **Qwen3-VL** | **55.4** |

**关键洞察**：
- 将 VAE 像素空间替换为 DINO 表征带来巨大提升（20.0% → 55.4%），**语义结构化潜在空间是有效规模化的关键**
- UWM 即使扩展到 1B 参数或替换为 MM-DiT，在 VAE 空间下仅边际改善（14.2% → 20.0%），说明表征瓶颈是根本性的
- MM-DiT 架构贡献 6.5%，模型规模从 0.5B 到 1B 贡献 4.7%

### 5.2 真实世界实验：夹爪操作

![图 6：夹爪操作成功率对比](https://arxiv.org/html/2602.12215v1/x6.png)

> **图 6**：真实世界夹爪操作任务成功率对比。所有模型在 Galbot 上少样本微调，评估 8 个任务。LDA 在 Pick & Place、Contact-rich、Fine 和 Long-horizon 操作上持续优于 GR00T-N1.6 和 π₀.₅。

| 任务类别 | 代表任务 | LDA vs 最强基线 |
|---------|---------|----------------|
| **Pick & Place** | Handover | 90% vs 70%（π₀.₅），50%（GR00T） |
| **Contact-rich** | Flip Box | 60% vs 20%（GR00T） |
| **Fine Manipulation** | Watering (pouring) | 80% vs 60%（π₀.₅） |
| **Long-horizon** | Clean Rubbish | **35% vs 0%**（两个基线均为 0%） |

**关键洞察**：长时程任务差距最大——Clean Rubbish 需要双臂协调、工具使用（簸箕）和顺序物体转移，LDA 达 35% 而基线完全失败，说明**显式动力学建模对长时程操作不可或缺**。

### 5.3 真实世界实验：灵巧操作

![图 7：灵巧操作成功率对比](https://arxiv.org/html/2602.12215v1/x7.png)

> **图 7**：灵巧操作成功率对比。3 个低自由度手（BrainCo）任务和 2 个高自由度手（Sharpa）任务。LDA（深蓝）持续优于基线，尤其在精细灵巧任务（拔钉子）和高自由度任务上。

| 任务 | 手类型 | LDA | π₀.₅ | GR00T |
|------|--------|-----|-------|-------|
| Pick Bottle | 低 DoF | **90%** | 20% | 75% |
| Pull Nail | 低 DoF | **80%** | 0% | 40% |
| Open Macbook | 低 DoF | 90% | 90% | 100% |
| Pick Bread | 高 DoF | **70%** | 10% | 20% |
| Flip Bread | 高 DoF | **90%** | 10% | 10% |

**关键洞察**：高自由度灵巧手任务差距最显著——Flip Bread 需要协调手指运动和连续接触推理，LDA 达 90% 而两个基线仅 10%。

### 5.4 泛化能力

| 条件 | π₀.₅ | GR00T | LDA |
|------|-------|-------|-----|
| 新物体 | 26.7 | 40.0 | **60.0** |
| 未见背景 | 20.0 | 40.0 | **60.0** |
| OOD 位置 | 6.7 | 20.0 | **40.0** |

LDA 在所有视觉和空间扰动下保持 60% 成功率，证明大规模潜在动力学预训练使模型忽略视觉干扰而聚焦任务关键可供性。

### 5.5 混合质量数据高效微调

| 任务 | 配置 | π₀.₅ | LDA |
|------|------|-------|-----|
| Place pen into box | 仅高质量 | 60 | 70 |
| | 高质量 + 低质量 | **40**（↓20）| **80**（↑10）|
| Bimanually remove lid | 仅高质量 | 50 | 50 |
| | 高质量 + 低质量 | **40**（↓10）| **60**（↑10）|

**关键洞察**：加入低质量轨迹后，π₀.₅ 显著退化（-10%~-20%），而 LDA 反而提升（+10%），证明通用数据摄取机制能有效利用噪声数据。

### 5.6 规模化分析

![图 10：LDA 规模化分析](https://arxiv.org/html/2602.12215v1/x10.png)

> **图 10**：LDA 规模化分析，以未见测试集上的动作预测误差评估。上：动作预测误差随 30k 小时训练数据降至 6.6。下：LDA 在模型规模 0.1B→1B 上持续优于 UWM，而基线快速饱和。

**三大发现**：

1. **通用数据摄取的有效性**：仅用 Policy Only 目标（灰线），增加数据量导致不稳定行为；完整共训练框架（蓝线）随异构数据增加持续改善，即使动作标注轨迹耗尽后，加入 10k 无动作视频仍继续降低预测误差

2. **DINO vs VAE 表征的规模化差异**：UWM 快速饱和，额外监督收益递减甚至为负——VAE 表征纠缠外观、几何和动力学，限制模型分解动作引起的状态转移；LDA 在语义结构化 DINO 空间中随模型容量、训练目标和数据多样性平滑扩展

3. **模型规模的有效性**：0.1B → 0.5B → 1B 参数在完整共训练框架下动作预测误差单调下降

### 5.7 动力学学习分析

#### 潜在前向动力学可视化

![图 9：潜在前向动力学可视化](https://arxiv.org/html/2602.12215v1/x9.png)

> **图 9**：潜向前向动力学可视化。模型生成准确的未来视觉表征（上），与 ground truth（下）在时间步上对齐，捕获语义物体结构和运动动力学。

模型产生尊重物理约束（物体持久性、接触连续性、运动一致性）的连贯未来状态预测，且聚焦任务相关物体而对视觉干扰不变。

#### 动作条件注意力

![图 11：动作条件注意力热力图](https://arxiv.org/html/2602.12215v1/x11.png)

> **图 11**：注意力热力图。"Push Right"（上）高亮杯子的前沿和轨迹；"Push Close"（下）聚焦接触面。模型排他性地关注可移动区域，忽略无关背景杂乱。

通过计算主动动作命令与 No-Op 命令的注意力差异 $\Delta A = |A_1 - A_2|$，隔离纯粹由动作条件引起的注意力变化。LDA 持续关注与命令交互因果相关的区域（接触面、力作用点、预期运动轨迹），而非静态外观。

---

## 核心洞察与总结评价

### 技术亮点

1. **通用数据摄取（Universal Data Ingestion）**：不是简单地"用更多数据"，而是让不同质量数据各尽其用——高质量学策略、低质量学动力学、无标注学视觉预测。这种角色感知设计使 30k+ 小时异构数据真正被有效利用。

2. **DINO 潜在空间替代 VAE 像素空间**：这是规模化成功的关键。VAE 表征纠缠外观、几何和动力学，UWM 在 VAE 空间下即使 1B 参数也仅 20% 成功率；DINO 空间保留物体级语义和空间一致性，使动力学学习随模型容量平滑扩展（20% → 55.4%）。

3. **MM-DiT 多专家架构**：动作和视觉各有模态特定 QKV/FFN（保留归纳偏置），通过共享自注意力交互，语言通过交叉注意力指导。这种设计在 1B 参数下稳定训练。

4. **任务嵌入 + 寄存器 Token**：优雅地实现单一模型内的多目标差异化训练，无需修改网络拓扑。

5. **混合质量数据反而有益**：低质量轨迹对 BC 方法有害（π₀.₅ 退化 10-20%），但对 LDA 有益（提升 10%），证明动力学建模能有效提取噪声数据中的监督信号。

### 局限性

1. **依赖固定 DINO 视觉特征**：DINO 编码器在预训练中冻结，无法针对下游任务自适应调整视觉表征，可能限制新视觉视角的泛化。
2. **主要依赖自我中心视角**：仅使用头戴式相机，缺少第三人称视角和多相机配置。
3. **动作空间统一的开销**：跨具身形态的手动坐标系对齐需要大量工程工作。
4. **长时程评估有限**：Clean Rubbish 虽然是长时程任务，但仅 35% 成功率，更复杂的长时程场景仍有挑战。
5. **仿真到真实的差距**：仿真数据占近 1/3，但仿真环境的物理保真度有限。

### 个人点评

LDA-1B 的核心贡献在于**将机器人基础模型的规模化瓶颈从"数据量"重新定义为"数据利用方式 + 表征空间"**。论文最有说服力的实验是规模化分析（Fig. 10）：UWM 在 VAE 空间下无论怎么加数据和参数都饱和，而 LDA 在 DINO 空间下平滑扩展——这清晰地表明，**表征质量比模型规模更重要**。

通用数据摄取的设计也非常实用：低质量数据不是负担而是资源，无标注视频不是废料而是免费监督。这种"物尽其用"的思路对实际部署意义重大——数据收集和标注是机器人学习最大的成本瓶颈。

**论文评分**：⭐⭐⭐⭐☆（4/5）
- 创新性：通用数据摄取 + DINO 潜在空间的组合有明确创新，规模化分析深入
- 完整性：仿真 + 真实世界实验充分，消融覆盖全面，动力学可视化有说服力
- 可复现性：EI-30K 数据集和代码承诺开源，数据处理管线详细披露

---

## Appendix 补充

### A. 机器人基础模型对比（Tab. I）

| 模型 | 数据源 | 数据量 | 动作质量 | 训练范式 | 参数量 |
|------|--------|--------|---------|---------|--------|
| π₀.₅ | Tele. | 10k+ | High | BC | 3B |
| RDT | Tele. | <10k | High | BC | 1B |
| GraspVLA | Sim. | 20k+ | High | BC | 2B |
| InternVLA-M1 | Sim. | <10k | High | BC | 3B |
| Being-H0 | Hum. | <10k | Mixed | Aln. + BC | 14B |
| InternVLA-A1 | Het. | 10k+ | High | VF + BC | 3B |
| GR00T-N1.6 | Het. | <10k | Mixed | LA + BC | 1B |
| UniVLA | Het. | <10k | Mixed | LA + BC | 7B |
| **LDA-1B** | **Het.** | **30k+** | **Mixed** | **UWM** | **1B** |

缩写：Tele.=遥操作, Sim.=仿真, Hum.=人类演示, Het.=异构数据, BC=行为克隆, VF=视觉前瞻, Aln.=对齐, LA=潜在动作建模, UWM=统一世界模型。

### B. RoboCasa-GR1 逐任务结果（Tab. VI）

24 个桌面重排和铰接物体操作任务的完整结果，LDA-1B 平均 55.4%，显著优于 UWM 变体（14.2%-20.0%）、GR00T-N1.6（47.6%）和 StarVLA（47.8%）。

### C. 真实世界任务配置

#### 夹爪操作任务（Galbot G1）

| 任务 | 描述 | 评估协议 |
|------|------|---------|
| Pick Vegetable | 拾取辣椒放入篮子 | 10 次试验；放入篮子为成功 |
| Handover | 左夹爪传递瓶子给右夹爪放入篮子 | 10 次试验 |
| Wipe Board | 用橡皮擦白板 | 10 次试验；0-2 分评分 |
| Flip Box | 双手翻转倒置储物盒 | 10 次试验 |
| Water Flower | 倒水浇花 | 10 次试验 |
| Knock Block | 用锤子敲击指定方块 | 60 次试验 |
| Sweep Table | 用扫帚将钉子扫入簸箕 | 10 次试验；按比例计分 |
| Throw Rubbish | 将纸团放入簸箕倒入垃圾桶 | 10 次试验；按比例计分 |

#### 灵巧手操作任务

| 任务 | 平台 | 手类型 | 评估协议 |
|------|------|--------|---------|
| Pick Bottle | Galbot + Sharpa | 22-DoF | 20 次试验 |
| Open MacBook | Unitree + BrainCo | 10-DoF | 20 次试验 |
| Pull Nail | Unitree + BrainCo | 10-DoF | 10 次试验；部分计分 |
| Pick Bread | Galbot + Sharpa | 22-DoF | 10 次试验 |
| Flip Bread | Galbot + Sharpa | 22-DoF | 10 次试验；首次翻转 1.0 分 |

### D. EI-30K 数据处理管线

1. **数据集标准化**：所有原始数据集转换为 LeRobot 2.1 格式，统一重采样至 10 Hz
2. **坐标对齐与清洗**：末端执行器坐标对齐、相机运动解耦、关键点标准化、数据验证
3. **后处理**：语言标注统一（VLM 辅助）、无意义手-物交互段移除、质量标签分配

### E. 动作条件注意力可视化方法

1. 提取中间 Transformer 块的注意力权重 $A_1$（活跃动作条件）
2. 生成基线注意力图 $A_2$（No-Op 静态命令）
3. 计算绝对差异 $\Delta A = |A_1 - A_2|$
4. 差异图隔离纯粹由动作引起的注意力变化，去除通用视觉显著性

---

## 参考信息

- **arXiv**：[https://arxiv.org/abs/2602.12215](https://arxiv.org/abs/2602.12215)
- **HTML**：[https://arxiv.org/html/2602.12215v1](https://arxiv.org/html/2602.12215v1)
- **Project Page**：[https://pku-epic.github.io/LDA](https://pku-epic.github.io/LDA)

---

*精读日期：2026-05-27*
