# OpenClaw-RL 深度解读笔记

> 论文：*OpenClaw-RL: Train Any Agent Simply by Talking*
> 作者：Yinjie Wang, Xuyang Chen, Xiaolong Jin, Mengdi Wang, Ling Yang（Princeton University）
> 代码仓库：https://github.com/Gen-Verse/OpenClaw-RL

---

## 目录

- [一、这篇工作到底在解决什么问题？](#一这篇工作到底在解决什么问题)
- [二、核心洞察：Next-State Signal](#二核心洞察next-state-signal)
- [三、系统架构：四组件全异步流水线](#三系统架构四组件全异步流水线)
- [四、方法一：Binary RL（二值奖励强化学习）](#四方法一binary-rl二值奖励强化学习)
- [五、方法二：OPD（Hindsight-Guided On-Policy Distillation）](#五方法二opd-hindsight-guided-on-policy-distillation)
- [六、方法三：Combine（组合方法）](#六方法三combine组合方法)
- [七、Track 2：通用 Agent RL](#七track-2通用-agent-rl)
- [八、实验结果与关键发现](#八实验结果与关键发现)
- [九、代码架构全景](#九代码架构全景)
- [十、从第一性原理的反思](#十从第一性原理的反思)

---

## 一、这篇工作到底在解决什么问题？

### 1.1 背景：AI Agent 的"数据浪费"

当你和一个 AI 助手对话时，每一次你的回复、每一次工具的执行结果、每一次终端的输出，都构成了对 AI 上一次行为的**隐式反馈**。例如：

- 用户说"不对，你应该先看一下文件再改"→ 意味着 AI 上一步做错了，而且给出了具体的改进方向
- 工具返回了一个错误 → 意味着 AI 的工具调用参数不正确
- 用户直接问下一个问题 → 暗示上一个回答是令人满意的

**然而，现有的所有系统都把这些信号白白扔掉了。** 它们仅仅将这些反馈作为下一轮对话的上下文，而没有用它来训练模型本身。

### 1.2 问题的本质

这个问题可以从第一性原理来理解。强化学习的核心要素是：

1. **状态 (State)**：当前的对话/环境上下文
2. **动作 (Action)**：模型生成的回复
3. **奖励 (Reward)**：对动作质量的评估
4. **转移 (Transition)**：环境对动作的响应，即"下一个状态"

在传统的 LLM RL 系统中（如 RLHF、GRPO），训练数据是**批量预收集**的——先采集一堆数据，再离线训练。但在实际部署中，AI 助手一直在**实时交互**，源源不断地产生数据。现有系统完全没有利用这个实时数据流。

### 1.3 OpenClaw-RL 的定位

OpenClaw-RL 提出的核心命题是：

> **每一次 Agent 交互都会产生一个 next-state signal（下一状态信号），这个信号可以作为免费的、实时的、在线的训练信号来源。**

而且这个信号是**通用的**：无论是个人对话、终端命令执行、GUI 操作、代码编写还是工具调用，都会产生 next-state signal。它们本质上是同一种东西，可以用同一个训练循环来处理。

---

## 二、核心洞察：Next-State Signal

### 2.1 两种信号类型

论文识别出 next-state signal 包含两种互补的信息：

#### （1）评价性信号（Evaluative Signal）

下一状态隐式地**打分**了上一个动作的好坏：

| 场景 | Next-State | 含义 |
|------|-----------|------|
| 用户重新提问/要求重做 | "不对，你重新来" | 上一步做错了（-1） |
| 用户继续问下一个问题 | "好的，那接下来..." | 上一步做对了（+1） |
| 工具返回成功结果 | `{"status": "ok"}` | 工具调用正确（+1） |
| 终端输出错误 | `Error: file not found` | 命令有误（-1） |
| 测试通过 | `All tests passed` | 代码修改正确（+1） |

这种信号可以被压缩为一个标量奖励 r ∈ {+1, -1, 0}。

#### （2）方向性信号（Directive Signal）

下一状态不仅告诉你"做错了"，还经常告诉你**应该怎么做**：

| 场景 | Next-State | 方向性信息 |
|------|-----------|-----------|
| 用户纠正 | "你应该先检查文件再编辑" | 具体的改进指令 |
| 编译错误 | `undefined variable 'x'` | 哪里出了问题 |
| SWE diff | 完整的代码差异 | 哪些 token 该改 |
| 用户反馈 | "别用那个库" | 要避免什么 |

这种信号比一个标量奖励丰富得多——它指出了**哪些 token 应该改变、应该怎样改变**。

### 2.2 为什么现有方法无法利用这些信号？

| 现有方法 | 局限性 |
|---------|--------|
| RLHF / DPO | 需要人工标注的偏好对，无法处理实时交互 |
| GRPO / DeepSeek-R1 | 批量离线训练，需要预收集数据集 |
| 标准蒸馏 | 需要一个单独的、更强的教师模型 |
| 在线 RL (veRL, AReal) | 假设批量收集模式，不支持实时多流交互 |

**OpenClaw-RL 的关键创新是：它同时回收了评价性信号和方向性信号，而且是在实时交互过程中完成的，不需要打断模型的正常服务。**

---

## 三、系统架构：四组件全异步流水线

### 3.1 整体架构

OpenClaw-RL 将整个系统解耦为**四个完全独立的异步组件**，彼此之间没有阻塞依赖：

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Policy Serving  │     │   Environment   │     │  Reward Judging │     │ Policy Training  │
│    (SGLang)      │ ──→ │  (HTTP / API)   │ ──→ │ (SGLang / API)  │ ──→ │   (Megatron)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

**为什么需要全异步？**

设想一个同步系统：模型生成回复 → 等待用户反馈 → PRM 打分 → 训练 → 更新权重 → 再服务下一个请求。在这种设计下：

- 用户必须等训练完成才能得到下一次回复（延迟极高）
- 如果多个用户同时使用，系统会完全阻塞
- 长时间的 rollout（如 SWE 任务可能需要数十步）会卡住整个流水线

全异步设计意味着：
- **模型继续服务请求**，不会因为训练而停止
- **PRM 在后台打分**，不阻塞服务或训练
- **训练在后台进行**，收集到足够样本后就开始
- **权重更新是 graceful 的**——替换而非重启

### 3.2 对应的代码实现

#### 组件 1：Policy Serving（策略推理服务）
- **技术栈**：SGLang（高性能 LLM 推理引擎）
- **代码位置**：由 slime 框架管理，通过 `--sglang-*` 参数配置
- **职责**：接收用户请求，生成回复，同时收集每个 token 的 log-probability

#### 组件 2：Environment（环境服务器）
- **个人 Agent**：用户设备通过 HTTP 连接，使用 OpenAI-compatible API
  - 代码：`openclaw-rl/openclaw_api_server.py` 中的 FastAPI 服务
  - 通过 `X-Session-Id`、`X-Turn-Type`、`X-Session-Done` 三个 HTTP header 追踪会话
- **通用 Agent**：云端环境（Docker 容器、虚拟机等）
  - Terminal：`terminal-rl/remote/pool_server.py`（Flask 服务，端口 18081）
  - GUI：`gui-rl/env_pool_server.py`（Flask 服务，端口 18080）
  - SWE：`swe-rl/server/swe_env_pool_server.py`（Flask 服务，端口 18090）

#### 组件 3：Reward Judging（奖励评估）
- **技术栈**：SGLang 或外部 API
- **代码**：
  - Binary RL 的 PRM 评估：`openclaw-rl/openclaw_api_server.py` 中的 `_prm_evaluate()`
  - OPD 的 hint 提取：`openclaw-opd/openclaw_opd_api_server.py` 中的 `_opd_evaluate()`
  - 通用 Agent 的 PRM：各自目录下的 PRM 相关代码

#### 组件 4：Policy Training（策略训练）
- **技术栈**：Megatron-LM（分布式训练框架）
- **代码**：`slime/train_async.py`（异步训练循环）
- **核心逻辑**：从 Data Buffer 拉取就绪的训练样本，计算 loss，更新权重

### 3.3 Session-Aware 环境服务器

对于个人 Agent，API 服务器需要理解多轮对话的结构。代码中使用三个 HTTP header 来实现：

```python
# openclaw_api_server.py 中的请求处理
x_session_id: str   # 会话 ID，标识一个完整的对话
x_turn_type: str     # "main"（主线轮次，可训练）或 "side"（旁线轮次，不训练）
x_session_done: bool # 会话是否结束
```

**Main-line turn vs Side turn 的区分非常关键：**

- **Main-line turn**：模型的主要回复和工具执行结果 → 生成训练数据
- **Side turn**：辅助查询、内存整理、环境转换等 → 不生成训练数据

这允许 RL 框架精确识别哪些轮次属于哪个会话，只在有意义的轮次上训练。

### 3.4 Graceful Weight Update

训练完成后，新权重需要替换到推理引擎中。这个过程是**不中断服务**的：

1. 训练器完成一个 step
2. 通知 rollout worker 暂停提交新样本（`pause_submission()`）
3. 权重通过分布式广播更新到 SGLang 引擎
4. 恢复提交（`resume_submission()`）
5. 清除旧的 record 文件，开始新一轮收集

整个过程中，如果有正在处理的用户请求，它们会正常完成，只是新请求会使用更新后的权重。

### 3.5 Non-Blocking Record

所有交互和奖励评估都被实时记录到 JSONL 文件中：

```python
# 记录内容包括：
# - 完整消息历史
# - prompt/response 文本
# - 工具调用
# - next-state 内容
# - PRM 评分（每个投票的得分和评估文本）
# - OPD hint 选择结果
```

记录写入是 fire-and-forget 的后台操作，不影响服务延迟。Record 文件在每次权重更新时清除，确保日志对应单一策略版本。

---

## 四、方法一：Binary RL（二值奖励强化学习）

### 4.1 核心思想

Binary RL 是最简单的方法：把 next-state signal 压缩为一个标量奖励 r ∈ {+1, -1, 0}，然后用类似 GRPO 的 PPO-style 方法训练。

### 4.2 PRM Judge 构建

给定模型的回复 `a_t` 和下一状态 `s_{t+1}`，PRM（Process Reward Model）对 `a_t` 的质量进行评判。

**PRM 的 System Prompt**（来自代码 `openclaw_api_server.py:75-117`）：

```
You are a process reward model (PRM) evaluating an AI assistant.
You will see the assistant's output and the subsequent user reply.
Judge the quality of the assistant's output based on the feedback.
Think step-by-step, then give your final score inside \boxed{}.
Valid scores: \boxed{1} (good), \boxed{-1} (bad), \boxed{0} (neutral).
```

PRM 会区分不同类型的 next-state：
- `role='user'`：用户的回复（可能包含满意/不满意的信号）
- `role='tool'`：工具返回值（成功执行通常是正面信号）

**评分规则**：
- `+1`（好）：任务按预期推进，用户继续下一步，工具返回成功
- `-1`（差）：用户要求重做/重试/修改，环境报错
- `0`（中性）：信息不足，无法判断

### 4.3 Majority Voting（多数投票）

为了增加评分的鲁棒性，对每个样本运行 `m` 次独立的 PRM 查询（默认 m=3），然后取多数投票：

```python
# openclaw_api_server.py:130-138
def _majority_vote(scores):
    counter = Counter(scores)
    max_count = max(counter.values())
    winners = [s for s, c in counter.items() if c == max_count]
    if len(winners) > 1:
        return 0.0   # 平票则返回中性
    return float(winners[0])
```

**为什么需要多数投票？** 因为 PRM 本身也是一个 LLM，它的判断可能不稳定。通过多次独立查询+投票，可以减少误判。

### 4.4 训练目标

最终的训练使用 PPO-style 的 clipped surrogate loss：

**重要性比率（Importance Ratio）：**
$$\rho_t = \frac{\pi_\theta(a_t | s_t)}{\pi_{\text{old}}(a_t | s_t)}$$

**损失函数：**
$$\mathcal{L}_{\text{pg}} = -\mathbb{E}_t\left[\min\left(\rho_t A_t, \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon_{\text{high}}) \cdot A_t\right)\right]$$

**总损失：**
$$\mathcal{L} = \mathcal{L}_{\text{pg}} + \beta_{\text{KL}} \cdot \mathcal{L}_{\text{KL}}$$

其中：
- `ε = 0.2`（下界裁剪）
- `ε_high = 0.28`（上界裁剪，非对称裁剪）
- `β_KL = 0.02`（KL 散度惩罚系数）
- `A_t = r_final`（优势值直接使用 PRM 的投票结果）

**注意与标准 GRPO 的区别**：由于这是实时对话场景，没有 group structure（一个 prompt 只有一个 response），所以**没有组内标准化**（disable-rewards-normalization）。

### 4.5 Loss Mask 和 At-Least-One 保证

不是所有的样本都会参与训练。代码中有一个精巧的 loss mask 机制：

```python
# openclaw_api_server.py:606-621
has_next_state = turn_data.get("has_next_state", False)
score = prm_result["score"] if prm_result else 0.0
exclude = not has_next_state or score == 0.0

# 如果还没有 next_state（最后一轮，没有反馈），排除
# 如果 PRM 评分为中性（0），也排除（信息不足以训练）
```

**At-Least-One 保证**：如果一个 session 的所有样本评分都是 0（中性），系统会强制保留至少一个有 next_state 的样本，避免整个 session 的数据被完全丢弃。

### 4.6 完整数据流

```
用户消息 → FastAPI 代理 → 转发到 SGLang 生成回复
                                    ↓
                          收集 per-token logprobs
                                    ↓
                          缓存 turn_data (prompt_ids, response_ids, logprobs)
                                    ↓
                       下一轮用户消息到达 → 提取 next_state
                                    ↓
                      异步触发 m 次 PRM 评估（并行）
                                    ↓
                              多数投票 → score
                                    ↓
                    创建 Sample（tokens, loss_mask, reward, logprobs）
                                    ↓
                           放入 output_queue
                                    ↓
              AsyncRolloutWorker 拉取 → 凑够 batch_size → 交给 SLIME Trainer
                                    ↓
                       计算 advantage → PPO loss → 更新权重
```

---

## 五、方法二：OPD（Hindsight-Guided On-Policy Distillation）

### 5.1 核心思想

Binary RL 把 next-state signal 压缩为一个标量 {+1, -1, 0}，这丢失了大量信息。当用户说"你应该先检查文件再编辑"时，这句话不仅表达了"你做错了"（评价性），还指出了"应该怎样做"（方向性）。

OPD 的核心洞察是：

> **如果我们把 next-state 中提取的改进提示（hint）加到原始 prompt 中，同一个模型就会生成不同的 token 分布——这个分布差异本身就是 token 级别的训练信号。**

换句话说：
- **Student**（学生）：基于原始 prompt 生成回复的模型
- **Teacher**（教师）：基于增强 prompt（原始 prompt + hint）"重新审视"同一回复的模型

**Teacher 和 Student 是同一个模型**——只是输入不同。这是 OPD 区别于标准蒸馏的关键：不需要单独的、更强的教师模型。

### 5.2 四步流程

#### Step 1：Hindsight Hint 提取

给定回复 `a_t` 和 next-state `s_{t+1}`，judge 模型同时做两件事：
1. 评分：+1（有有用的 hindsight）或 -1（没有）
2. 如果 +1，提取一个简洁的、可操作的 hint（1-3 句话）

**Judge 的 System Prompt**（来自代码 `openclaw_opd_api_server.py`）：

```
You are a process reward model used for hindsight hint extraction.
Decide whether the next state reveals useful hindsight that could have
improved the assistant response at turn t.
- Output \boxed{1} if yes; provide a hint in [HINT_START]...[HINT_END].
- Output \boxed{-1} if no; do not provide a hint.
- Hint must be concrete and actionable (1-3 sentences).
```

**关键设计决策：不直接使用 `s_{t+1}` 作为 hint。** 原始的 next-state 信号通常很嘈杂、冗长，或者包含不相关的信息（比如用户的回复可能同时包含纠正和一个新问题）。Judge 模型负责**蒸馏**出关键的方向性信息。

#### Step 2：Hint 选择和质量过滤

运行 `m` 次并行 judge 评估后：

```python
# openclaw_opd_api_server.py
# 过滤标准：
# 1. 评分必须是 +1
# 2. hint 必须 > 10 个字符（排除空洞的 hint）
# 3. 在所有合格的 hint 中，选最长的（最信息量丰富的）

valid_hints = [h for score, h in votes if score == +1 and len(h) > 10]
if not valid_hints:
    DROP_SAMPLE()  # 没有有效 hint，放弃这个样本
hint = max(valid_hints, key=len)  # 选最长的
```

**这意味着 OPD 会丢弃很多样本**——只有那些 next-state 中确实包含清晰改进方向的样本才会参与训练。这是一个**质量 vs 数量**的权衡：Binary RL 覆盖所有评分过的轮次，OPD 只在有方向性信号的轮次上提供高精度 token 级别指导。

#### Step 3：Enhanced Teacher 构建

将选出的 hint 附加到最后一条用户消息中：

```python
# 原始用户消息
"请帮我写一个排序函数"

# 增强后的用户消息
"请帮我写一个排序函数\n\n[user's hint / instruction]\n你应该使用快速排序而不是冒泡排序，因为数据量很大"
```

这创造了一个"如果用户一开始就给了这个提示"的假设情境。在这个增强 prompt 下，模型"知道"正确的方向，因此它对原始回复中每个 token 的概率评估会发生变化。

#### Step 4：Token-Level Advantage 计算

用增强 prompt 查询模型，得到 teacher 对原始回复 `a_t` 每个 token 的 log-probability：

$$A_t[k] = \log \pi_{\text{teacher}}(a_t[k] \mid s_{\text{enhanced}}) - \log \pi_\theta(a_t[k] \mid s_t)$$

对于每个 token `k`：
- **A_t[k] > 0**：teacher（知道 hint）认为这个 token 应该被加强 → 这个 token 是好的
- **A_t[k] < 0**：teacher 认为这个 token 不太合适 → 这个 token 应该被削弱

这就实现了**per-token 方向性指导**：在同一个回复内，有些 token 被强化，有些被抑制。这比 Binary RL 的"整个回复统一奖励 +1/-1"精细得多。

### 5.3 Top-K 蒸馏变体

除了 token-level advantage（只关注"原始回复中每个 token 应该增还是减"），还有一个 Top-K 变体：

```python
# topk_distillation_loss.py
# 获取 teacher 在每个位置的 top-K token 分布
# 构建 K+1 个 bin：K 个 top token + 1 个 tail（剩余概率质量）
# 计算 student 和 teacher 在这 K+1 个 bin 上的 reverse KL 散度
```

这个方法更强——它不只告诉 student "这个 token 应该增减"，而是告诉 student "在这个位置，概率分布应该长什么样"。

### 5.4 OPD 的代码结构

```
openclaw-opd/
├── openclaw_opd_api_server.py   # API 服务器 + hint 提取 + teacher 查询
│   ├── _build_hint_judge_messages()   # 构建 judge prompt
│   ├── _parse_judge_result()          # 解析 \boxed{} 和 [HINT_START]...[HINT_END]
│   ├── _select_best_hint()            # 选择最佳 hint
│   ├── _append_hint_to_messages()     # 将 hint 附加到用户消息
│   ├── _compute_teacher_log_probs()   # 查询 teacher log-probs
│   └── _opd_evaluate()               # 完整的 OPD 评估流程
├── openclaw_opd_rollout.py      # 连接 API 服务器和 SLIME trainer
├── topk_distillation_loss.py    # Top-K 蒸馏损失函数
├── run_qwen3_4b_openclaw_opd.sh         # Token-level OPD
├── run_qwen3_4b_openclaw_opd_topk.sh    # Top-K 蒸馏
└── run_qwen3_4b_openclaw_opd_topk_lora.sh  # Top-K + LoRA
```

---

## 六、方法三：Combine（组合方法）

### 6.1 为什么要组合？

Binary RL 和 OPD 是**互补的**，不是竞争的：

| 维度 | Binary RL | OPD |
|------|-----------|-----|
| 信号类型 | 评价性（好/坏） | 方向性（应该怎样做） |
| 优势值粒度 | 整个序列一个标量 | 每个 token 一个值 |
| 覆盖率 | 所有评分过的轮次 | 仅有有效 hint 的轮次 |
| 反馈类型 | 隐式（用户继续/重做）、环境输出 | 显式（用户纠正、详细错误信息） |
| 信号丰富度 | 每样本 1 个标量 | 每 token 1 个值 |

- Binary RL 的问题：信号太粗——"+1"不告诉你哪些 token 好，"-1"不告诉你哪里错
- OPD 的问题：覆盖率太低——很多 next-state 没有清晰的方向性信息（用户只是简单地继续对话）

**组合方法让两者取长补短：Binary RL 提供广覆盖的粗信号，OPD 在可用时提供精细信号。**

### 6.2 组合方式

由于两种方法使用**相同的 PPO loss**，只是 advantage 的计算不同，所以组合非常优雅——直接加权相加 advantage：

$$A_t = w_{\text{binary}} \cdot r_{\text{final}} + w_{\text{opd}} \cdot \left(\log \pi_{\text{teacher}}(a_t | s_{\text{enhanced}}) - \log \pi_\theta(a_t | s_t)\right)$$

其中 `w_binary = w_opd = 1` 是默认值。

### 6.3 巧妙的样本调度

组合方法的核心实现在 `openclaw-combine/openclaw_combine_api_server.py`。它对每个 turn 生成**一个样本**，但根据信号可用性分为三种情况：

| Hint 被接受 | PRM 有评分 | 样本类型 | reward | teacher_logprobs |
|:---:|:---:|:---:|:---:|:---:|
| ✓ | ✓ | OPD+RL | ±1 | 真实 teacher logprobs |
| ✓ | ✗ | OPD-only | 0 | 真实 teacher logprobs |
| ✗ | ✓ | RL-only | ±1 | 设为 rollout logprobs |
| ✗ | ✗ | 丢弃 | - | - |

**这个设计的精妙之处在于：通过设置特定的默认值，让不适用的分支自然"归零"：**

- **OPD-only 样本**：reward = 0 → GRPO advantage = 0 → Binary RL 分支不起作用
- **RL-only 样本**：teacher_logprobs = rollout_logprobs → OPD advantage = 0 → OPD 分支不起作用
- **OPD+RL 样本**：两个分支都有真实值 → 两者都起作用

这意味着**不需要任何 if-else 分支**——在 loss 函数中统一处理所有样本，每种类型自然只激活相关的分支。

### 6.4 组合 Loss 函数

```python
# combine_loss.py 的核心逻辑
def combine_loss_function(batch, args):
    # 1. 计算 GRPO advantage（从 reward 来）
    grpo_advantages = batch["rewards"]  # 广播到所有 token

    # 2. 获取 old log-probs
    old_logprobs = batch["rollout_log_probs"]

    # 3. 获取 teacher log-probs
    teacher_logprobs = batch["teacher_log_probs"]

    # 4. 组合 advantage
    w_rl = float(os.getenv("OPENCLAW_COMBINE_W_RL", "1.0"))
    w_opd = float(os.getenv("OPENCLAW_COMBINE_W_OPD", "1.0"))

    combined_adv = w_opd * (teacher_logprobs - old_logprobs) + w_rl * grpo_advantages

    # 5. 标准 PPO clipped surrogate loss
    ratio = exp(new_logprobs - old_logprobs)
    loss = -min(ratio * combined_adv, clip(ratio, 1-ε, 1+ε_high) * combined_adv)
```

---

## 七、Track 2：通用 Agent RL

OpenClaw-RL 不只是个人助手优化工具——同样的异步 RL 架构也支持大规模通用 Agent 训练。

### 7.1 四种环境

| 环境 | 代码目录 | 观察空间 | 动作空间 | Next-State Signal |
|------|---------|---------|---------|------------------|
| Terminal | `terminal-rl/` | 终端输出 | Bash 命令 | stdout/stderr, exit code |
| GUI | `gui-rl/` | 截图（1920×1080） | PyAutoGUI 操作 | 视觉状态变化、任务进度 |
| SWE | `swe-rl/` | 代码仓库 + 测试套件 | 编辑命令、Bash | 测试结果、diff、lint 输出 |
| Tool-call | `toolcall-rl/` | 数学问题 | Python 代码执行 | 返回值、错误信息 |

### 7.2 Terminal Agent（终端 Agent）

**核心文件**：`terminal-rl/generate.py`

**架构**：
- 使用 CAMEL 框架的 Agent（支持多轮工具调用）
- 远程 Docker 容器作为终端环境
- Pool Server 管理任务队列和容器分配

**工作流**：
1. 从 Pool Server 获取一个任务和一个远程环境
2. Agent 接收任务描述
3. Agent 生成 bash 命令 → 在远程容器中执行 → 获取 stdout/stderr
4. 重复直到完成或达到最大步数
5. 评估任务完成度 → 生成训练样本

**PRM 集成**：可选的 PRM Agent 对每一步进行评分（step-wise reward），而非只在最终给出 outcome reward。

### 7.3 GUI Agent（图形界面 Agent）

**核心文件**：`gui-rl/generate_with_gui.py`

**架构**：
- 使用 Qwen3-VL（视觉语言模型）
- 云端虚拟机（Volcengine/AWS/Aliyun）运行 Ubuntu 桌面
- OSWorld 评估器验证任务完成度

**工作流**：
1. 从 Pool Server 获取 VM 实例
2. 截屏 → 发送给 VLM → 解析为 PyAutoGUI 动作
3. 执行动作（鼠标点击、键盘输入等）
4. 重复直到完成或达到最大步数（默认 30 步）
5. 通过 OSWorld evaluator 评估

**视觉 PRM**：可选的视觉 reward agent，通过比较"动作意图"和"下一帧截图"来评估每步质量。

### 7.4 SWE Agent（软件工程 Agent）

**核心文件**：`swe-rl/generate_with_swe_remote.py`

**架构**：
- Mini-SWE-Agent（YAML 配置的多轮 bash 编排器）
- 远程 Docker 容器（SWE-Bench 实例）
- 上下文管理器：head(30%) + tail(70%) 策略防止溢出

**工作流**：
1. 获取 GitHub issue 描述和代码仓库
2. Agent 生成 THOUGHT + bash 命令
3. 在远程容器中执行
4. 解析输出，继续下一步
5. 提交 patch → 在全新容器中运行测试套件评估

**上下文管理**：SWE 任务通常非常长（几十步），容易超出上下文窗口。`swe_context_manager.py` 使用 head+tail 策略：保留前 30% 的早期探索和后 70% 的近期历史，确保训练时看到的上下文和推理时完全一致。

### 7.5 Tool-call Agent（工具调用 Agent）

**核心文件**：`toolcall-rl/generate_with_retool.py`

**架构**：
- Qwen3/Qwen2.5 + Python 代码执行工具
- 安全沙箱（白名单模块，120 秒超时，4GB 内存限制）
- 数学问题求解（DAPO-Math-17k 训练，AIME-2024 评估）

**安全沙箱设计**（`tool_sandbox.py`）：
```python
# 允许的模块：math, random, datetime, collections, itertools, ...
# 禁止的模式：os, sys, subprocess, eval, exec, open, __import__, ...
# 限制：超时 120 秒，内存 4GB，最大并发 32 进程
```

### 7.6 Step-wise Reward 的重要性

通用 Agent 任务通常是长 horizon 的（几十步）。如果只在最终给出 outcome reward（成功/失败），那么前面几十步的梯度信号非常稀疏——模型很难知道哪一步做对了、哪一步做错了。

OpenClaw-RL 在通用 Agent 场景中集成了 **step-wise process reward**：

$$r_t = o + \frac{1}{m}\sum_{i=1}^{m} \text{PRM}(a_t, s_{t+1})_i$$

其中 `o` 是最终的 outcome reward，`PRM(a_t, s_{t+1})` 是基于 next-state signal 的 step-wise 评分。

**Step-wise Standardization**：由于有了 step-wise reward，不能像 GRPO 那样简单地做组内标准化（因为同组内不同 step 的 reward 不同）。OpenClaw-RL 采用按 step index 分组标准化——把所有轨迹中"第 k 步"的 reward 放在一起标准化。

---

## 八、实验结果与关键发现

### 8.1 个人 Agent：学生和教师场景

**实验设置**：
- **学生场景**：用 LLM 模拟一个用 OpenClaw 做作业的学生，不想被发现在用 AI（偏好自然、非 AI 风格的回复）
- **教师场景**：用 LLM 模拟一个用 OpenClaw 批改作业的教师，希望评语具体且友好
- **模型**：Qwen3-4B
- **数据**：GSM8K 数学题

**评估方法**：用相同的 LLM 模拟器对 OpenClaw 的第一次回复进行个性化评分（0-1 分）。

### 8.2 关键实验结果

#### Q1：Binary RL vs OPD vs Combined

| 方法 | 8 步后分数 | 16 步后分数 |
|------|-----------|-----------|
| Base | 0.17 | 0.17 |
| Binary RL | 0.25 | 0.23 |
| OPD | 0.25 | 0.72 |
| **Combined** | **0.76** | **0.81** |

**关键发现**：
1. **Binary RL 单独使用效果有限**——从 0.17 提升到 0.25 后就趋于平稳甚至下降。原因：标量奖励信号太粗，难以持续指导优化。
2. **OPD 有延迟效应**——8 步时和 Binary RL 持平（0.25），但 16 步时飙升到 0.72。原因：OPD 的 hint 过滤很严格，早期样本稀疏，但每个样本的信息量大，长期效果显著。
3. **组合方法远超两者之和**——8 步就达到 0.76，16 步达到 0.81。Binary RL 提供快速、广覆盖的初始信号，OPD 提供精细调整。

#### Q2：个性化效果的案例分析

**学生场景（优化前 vs 优化后）**：

优化前（0.17 分）：
```
The handbag cost $220. Here's how:
1. Shoes cost $80
2. 3 times the shoe cost = 3 × $80 = $240
3. Subtract $20: $240 - $20 = **$220**
**Final Answer**: $220    ← 典型的 AI 风格：加粗、分步骤、格式化
```

优化后（0.76 分）：
```
Jillian's handbag costs $220. Here's the step-by-step solution:
1. 3 times the shoe cost: 3 * $80 = $240.
2. Subtract $20: $240 - $20 = $220. ✅  ← 更自然、更像人写的
```

Agent 学会了避免过于 AI 化的表达（加粗、`**`标记、过度结构化），转向更自然的风格。

**教师场景（优化前 vs 优化后）**：

优化前：`Correct. Well done!`（冷淡、简短）

优化后：
```
Your step-by-step approach is fantastic! You correctly added 20 + 44 to get 64,
then calculated 100 - 64 = 36. This shows you understand how to break down
the problem and find the solution. Well done!
```

Agent 学会了给出具体的、有温度的评语，而不是干巴巴的"正确"。

#### Q3：通用 Agent 跨场景一致性

| 环境 | 模型 | 训练步数 | 提升 |
|------|------|---------|------|
| Terminal | Qwen3-8B | 100 步 | 0.20 → 0.48 |
| GUI | Qwen3VL-8B-Thinking | 100 步 | 0.26 → 0.33 |
| SWE | Qwen3-32B | 30 步 | 0.05 → 0.175 |
| Tool-call | Qwen3-4B-SFT | 250 步 | 0.08 → 0.16 |

所有四种环境都展示了一致的提升，验证了 next-state signal 作为统一训练信号源的有效性。

#### Q4：Process Reward 的价值

| 环境 | Integrated (PRM+Outcome) | Outcome only |
|------|:---:|:---:|
| Tool-call | **0.30** | 0.17 |
| GUI | **0.33** | 0.31 |

集成 step-wise PRM 和 outcome reward 显著优于仅使用 outcome reward，尤其在 tool-call 场景（0.30 vs 0.17）。

---

## 九、代码架构全景

### 9.1 目录结构

```
OpenClaw-RL/
├── openclaw-rl/                 # 方法一：Binary RL
│   ├── openclaw_api_server.py   # FastAPI 代理 + PRM 评分 + 样本提交（731 行）
│   ├── openclaw_rollout.py      # 异步 rollout worker（153 行）
│   ├── run_qwen3_4b_openclaw_rl.sh      # 全量训练脚本
│   └── run_qwen3_4b_openclaw_rl_lora.sh # LoRA 训练脚本
│
├── openclaw-opd/                # 方法二：OPD
│   ├── openclaw_opd_api_server.py   # hint 提取 + teacher 查询（1001 行）
│   ├── openclaw_opd_rollout.py      # rollout worker
│   ├── topk_distillation_loss.py    # Top-K 蒸馏损失函数
│   ├── run_qwen3_4b_openclaw_opd.sh         # Token-level OPD
│   ├── run_qwen3_4b_openclaw_opd_topk.sh    # Top-K 蒸馏
│   └── run_qwen3_4b_openclaw_opd_topk_lora.sh
│
├── openclaw-combine/            # 方法三：组合
│   ├── openclaw_combine_api_server.py  # 三路样本调度
│   ├── openclaw_combine_rollout.py     # rollout worker
│   ├── combine_loss.py                 # 加权组合 loss
│   ├── run_qwen3_4b_openclaw_combine.sh       # 全量训练
│   └── run_qwen3_4b_openclaw_combine_lora.sh  # LoRA
│
├── terminal-rl/                 # 终端 Agent RL
│   ├── generate.py              # RL 循环主入口
│   ├── agent_runner.py          # Agent 编排
│   ├── agent/                   # CAMEL Agent + PRM Agent
│   └── remote/                  # Pool Server + 远程环境
│
├── gui-rl/                      # GUI Agent RL
│   ├── generate_with_gui.py     # 轨迹生成
│   ├── env_pool_server.py       # 虚拟机池管理
│   ├── agents/                  # Qwen3VL Agent + Reward Agent
│   └── desktop_env/             # OSWorld 环境包装
│
├── swe-rl/                      # SWE Agent RL
│   ├── generate_with_swe_remote.py  # RL 循环
│   ├── swe_env_client.py        # HTTP 客户端
│   ├── swe_context_manager.py   # 上下文窗口管理
│   ├── swe_prm.py               # PRM 评分
│   └── server/                  # Docker 节点服务 + 负载均衡
│
├── toolcall-rl/                 # 工具调用 Agent RL
│   ├── generate_with_retool.py  # 工具调用 RL
│   ├── tool_sandbox.py          # Python 安全沙箱
│   └── *_data_processing.py     # 数据预处理
│
├── openclaw-test/               # 端到端评估
│   ├── student_chat.py          # 学生做作业模拟
│   └── teacher_chat.py          # 教师批改作业模拟
│
├── openclaw-tinker/             # Tinker 云端训练（无需 GPU）
│   ├── run.py                   # CLI 入口
│   ├── trainer.py               # 训练循环
│   ├── api_server.py            # 三种方法的 API 服务器
│   ├── scorers.py               # 评分器
│   └── data_formatter.py        # 数据转换
│
├── slime/                       # 底层 RL 训练框架
│   ├── train.py / train_async.py    # 同步/异步训练循环
│   ├── slime/ray/rollout.py         # RolloutManager（1500+ 行）
│   ├── slime/ray/actor_group.py     # 分布式训练 Actor 管理
│   ├── slime/backends/megatron_utils/  # Megatron 集成
│   ├── slime/rollout/sglang_rollout.py # SGLang 推理
│   └── slime/utils/arguments.py     # 参数系统（~3000 行）
│
├── Megatron-LM/                 # Megatron 分布式训练引擎
├── openclaw/                    # OpenClaw 个人助手客户端
└── instructions/                # 环境配置指南
```

### 9.2 GPU 分配策略

**默认 8 GPU 配置（全量训练）**：

| 组件 | GPU 数量 | 用途 |
|------|---------|------|
| Actor (Megatron) | 4 | 策略训练 |
| Rollout (SGLang) | 2 | 模型推理服务 |
| PRM (SGLang/API) | 2 | 奖励模型/judge |

**4 GPU LoRA 配置**：

| 组件 | GPU 数量 | 用途 |
|------|---------|------|
| Actor (FSDP) | 2 | LoRA 训练 |
| Rollout (SGLang) | 1 | 模型推理 |
| PRM | 1 | 奖励模型 |

**Tinker 配置**：0 GPU（全部在云端）

### 9.3 关键超参数总结

| 参数 | 个人 Agent | 通用 Agent | 含义 |
|------|-----------|-----------|------|
| learning rate | 1e-5 | 1e-6 | 学习率（个人 Agent 更大，因为数据更少） |
| KL loss coef | 0.02 | 0.01 | KL 散度惩罚（防止策略偏移太远） |
| eps-clip | 0.2 | 0.2 | PPO 下界裁剪 |
| eps-clip-high | 0.28 | 0.28 | PPO 上界裁剪（非对称） |
| rollout-batch-size | 16 | 4-16 | 每次训练的样本数 |
| PRM M | 1-3 | 1-3 | PRM 投票次数 |
| max-response-len | 8192 | 4096-8192 | 最大回复长度 |
| temperature | 0.6 | 0.6 | 采样温度 |

---

## 十、从第一性原理的反思

### 10.1 这篇工作最根本的贡献是什么？

**不是一个新的 loss 函数，而是一个新的数据来源。**

所有现有的 LLM RL 系统——RLHF、DPO、GRPO、DAPO——都假设训练数据是预先收集好的。OpenClaw-RL 的根本创新在于指出：**模型在正常服务过程中就在源源不断地生成训练数据**，而且这些数据是免费的（next-state signal）。

这类似于自然界的学习方式：你不需要停下来"训练"——你在做事的过程中就在学习。你说了一句话，对方的反应就是你的即时反馈。

### 10.2 OPD 为什么是一个有意义的创新？

OPD 解决了一个深层次的问题：**如何从文本反馈中提取 token 级别的梯度信号？**

传统方法面对用户纠正"你应该先检查文件"时，最多能做的是给整个回复一个 -1 的奖励。但回复中可能有些部分是对的（比如理解了需求），有些是错的（比如跳过了检查步骤）。标量奖励无法区分这两部分。

OPD 的巧妙之处在于利用了一个**自然实验**：
- 控制组：模型在原始 prompt 下的 token 分布
- 实验组：模型在"原始 prompt + 改进提示"下的 token 分布
- 差异：哪些 token 在知道答案后应该被加强/削弱

这避免了需要一个单独的强教师模型——**模型既是学生又是教师**，只是输入信息不同。

### 10.3 异步架构为什么是必要的？

很多人可能觉得"全异步"只是一个工程优化。实际上，它是**功能性需求**：

1. **用户体验**：如果模型服务因为训练而停顿，用户会放弃使用
2. **数据多样性**：异步允许多个交互流同时贡献训练数据
3. **长 horizon 支持**：SWE 任务可能运行几十分钟。同步设计意味着整个系统在此期间完全停滞
4. **持续学习**：模型需要在不断接收新数据的同时进行训练，而非"收集→停止→训练→重启"的循环

### 10.4 局限性与开放问题

1. **PRM 的可靠性**：整个系统依赖 PRM/Judge 的评分质量。如果 PRM 本身有偏差（比如总是给高分），训练信号就会退化。论文通过 majority voting 缓解，但没有从根本上解决。

2. **冷启动问题**：模型需要先产生一些交互才能开始优化。如果初始模型太差，用户可能在看到任何改善之前就放弃了。

3. **灾难性遗忘**：持续在特定用户的数据上训练，可能导致模型在其他任务上性能下降。论文使用 KL 散度惩罚来缓解，但这是一个根本性的权衡。

4. **评估困难**：如何客观评估"个性化"的质量？论文使用 LLM 模拟器，但真实用户的行为远比模拟器复杂。

5. **隐私与安全**：模型在用户数据上实时训练，需要确保数据不泄露。论文强调了 self-hosted 的设计，但实际部署中仍需要更多安全措施。

### 10.5 这项工作对 AI Agent 发展的启示

OpenClaw-RL 展示了一个重要的方向：**AI Agent 不应该是静态的——它应该在使用过程中不断进化。**

这暗示了一种新的 AI 产品范式：
- **传统范式**：训练 → 部署 → 用户使用（模型固定）
- **OpenClaw 范式**：训练 → 部署 → 用户使用 → 模型在后台学习 → 模型变得更好 → 用户使用得更多 → ...

这形成了一个**正反馈循环**——模型越用越好，用户越用越多，数据越来越多，模型更好。这可能是通向真正"个人 AI 助手"的关键路径之一。
