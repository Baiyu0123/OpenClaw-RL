# OpenClaw-RL 详细学习笔记

## 0. 这份笔记在讲什么

这份笔记不是简单复述 README，也不是只摘论文摘要。

我做的是下面这件事：

1. 先从第一性原理出发，问这个项目到底想解决什么问题。
2. 再把论文里的核心方法拆开讲清楚。
3. 然后把仓库里真正决定训练行为的核心代码路径逐个对上。
4. 最后指出论文和代码之间哪些地方完全一致，哪些地方是工程化补丁，哪些地方是值得警惕的默认假设。

需要先说明一个范围问题：

- 这个仓库除了 OpenClaw-RL 自身代码，还打包了若干上游/外部项目，例如 `slime/`、`Megatron-LM/`、`openclaw/`、`swe-rl/mini-swe-agent/`。
- 如果机械地“逐行讲所有文件”，会把大量篇幅浪费在上游基础设施、任务样例 JSON、文档资源和第三方实现上。
- 所以我的处理方式是：**完整梳理 OpenClaw-RL 自身的关键实现，并深入阅读它与 `slime` / `Megatron-LM` 的接口层代码**，因为真正决定论文方法如何落地、样本如何进入训练、reward 如何变成 advantage 的，就是这些文件。

如果你只想先抓全局，先看第 1、2、3、4、11 节。
如果你想真正搞懂“论文里的每一句方法在代码里怎么落地”，重点看第 5、6、7、8、9 节。

---

## 1. 先从第一性原理：OpenClaw-RL 到底想解决什么问题？

### 1.1 传统 LLM 强化学习在 agent 场景里的根本问题

传统 RL for LLM，尤其是 RLVR（reinforcement learning with verifiable rewards）或类似 GRPO 的做法，通常依赖这样的监督：

- 给模型一个任务
- 让模型输出整段答案或整条轨迹
- 在最后只给一个结果分数

这套方法对“短答案、可验证、终局奖励明确”的问题很合适，比如数学题最终答对没答对。

但 agent 不是这样工作的。

一个 agent 的典型过程是：

1. 看到当前状态
2. 输出一步动作或一句回复
3. 环境立刻返回新状态
4. 再输出下一步
5. 如此反复很多轮

在这个过程中，真正有信息量的不是只有“最后成功/失败”，而是**每一步动作之后，环境给出的下一状态**。

例如：

- 聊天 agent：用户下一句回复就是反馈
- terminal agent：命令的 stdout/stderr 和 exit code 就是反馈
- GUI agent：执行动作后的新截图就是反馈
- SWE agent：测试结果、报错、diff、lint 输出就是反馈
- tool-call agent：工具返回值和错误 trace 就是反馈

也就是说，**agent 的每一步之后，天然都会得到一个“下一状态信号”**。

OpenClaw-RL 的核心洞察就是：

> 不要只盯着最终 outcome。每一步动作之后的 next-state，本身就是训练信号，而且这种信号在聊天、终端、GUI、SWE、工具调用等场景里是统一存在的。

### 1.2 论文想做的不是“发明一个新 loss”，而是统一一种训练视角

这篇论文最重要的地方，不是提出了一个特别复杂的新 optimizer，而是提出了一个统一视角：

- 把所有 agent 交互都抽象成：
  - 状态 `s_t`
  - 动作 `a_t`
  - 下一状态 `s_{t+1}`
- 然后从 `s_{t+1}` 里恢复训练信号

这就把不同 agent 场景统一起来了。

换句话说，论文的真正主张不是：

- “OpenClaw 这个聊天 agent 很特别”

而是：

- “任何 agent，只要它每做一步都会收到下一状态，那就都可以被这个框架训练”

这也是仓库为什么同时包含：

- `openclaw-rl/`
- `terminal-rl/`
- `gui-rl/`
- `swe-rl/`
- `toolcall-rl/`

因为作者想证明：这不是一个只对“个人助手聊天”有效的技巧，而是一个**统一的 agent RL 框架**。

---

## 2. 用一句话概括整篇论文

**OpenClaw-RL 的核心思想是：把 agent 每一步动作之后产生的 next-state 当作通用反馈源；如果这个反馈只表达“好/坏”，就做 Binary RL；如果它还包含“应该怎么改”的方向性信息，就做 OPD；如果是长时程 agent 任务，就把每步过程奖励和最终 outcome 奖励整合起来。**

再压缩一点：

- next-state 是信息源
- PRM/judge 负责把它变成可训练信号
- trainer 负责把信号变成梯度
- serving 不停机，训练后台持续进行

---

## 3. 论文的核心方法，先讲人话，再讲公式

## 3.1 问题形式化

论文第 2 节把所有交互流统一为一个 MDP：

- 状态 `s_t`：到当前为止的完整上下文
- 动作 `a_t`：模型在该状态下生成的一段 token
- 转移：执行动作后环境产生 `s_{t+1}`
- 奖励：由 `s_{t+1}` 推断出来

这里最关键的是最后一项：

> reward 不是直接写死在环境里的，而是从 next-state 中恢复出来的。

这点很重要，因为很多真实 agent 环境没有一个现成的 dense reward API。
但是它们几乎都有 next-state：

- 用户抱怨了
- 命令失败了
- 截图没变化
- 测试没过

这些都不是标量 reward，但都包含 reward 信息。

---

## 3.2 方法一：Binary RL

### 3.2.1 它在解决什么问题？

有时候 next-state 只告诉你一件事：

- 这一步大体上是对的
- 或者这一步大体上是错的

例如：

- 用户说“不是这个意思”
- shell 报错
- GUI 没有产生任何进展

这种反馈未必能告诉你“具体该改哪些 token”，但它至少能给一个粗粒度判断。

于是作者引入 Binary RL：

- 用 PRM/judge 读取 `(a_t, s_{t+1})`
- 输出 `+1 / -1 / 0`
- 再把这个标量广播给整段 response

### 3.2.2 论文中的公式含义

Binary RL 的 advantage 很简单：

\[
A_t = r_{final}
\]

其中 `r_final` 是多次 judge 投票后的最终结果。

然后它进入一个 PPO-style clipped objective。

这里作者强调一点：

- 这不是标准的 GRPO 组内标准化设置
- 因为聊天/在线 agent 场景天然没有“同题多采样的组结构”

所以个人 agent 路径里，作者本质上做的是：

- sequence-level scalar reward
- PPO 风格裁剪
- 但 advantage 本身非常简单，就是这个 scalar

### 3.2.3 直观理解

把它想成一句话就够了：

> 如果下一状态整体表明你刚才这步不错，那就把刚才这整段回答整体上推高；如果整体表明不行，就整体压低。

优点：

- 通用
- 稳定
- 样本覆盖广

缺点：

- 粗
- 一整段 token 被同一个标量驱动
- 不知道到底哪几个 token 才是错的

---

## 3.3 方法二：OPD（Hindsight-Guided On-Policy Distillation）

### 3.3.1 它在解决 Binary RL 的哪一个短板？

Binary RL 最大的问题是：

- 它知道这一步“好/坏”
- 但不知道这一步“具体该怎么改”

而真实反馈里，经常会出现方向性信息。

例如：

- “你应该先检查文件再编辑”
- “不要用这个库”
- “这里点错按钮了”
- “这个报错说明路径不对”

这些信息不是单纯的 reward，而是**纠错提示**。

作者认为：这种提示如果只压成一个 `+1/-1`，信息损失太大。

于是提出 OPD。

### 3.3.2 OPD 的核心思路

OPD 的逻辑是：

1. 先看 next-state 有没有“可提炼的 hindsight hint”
2. 如果有，用 judge 提炼成简短、明确、可执行的提示
3. 把这个 hint 拼回原始 prompt
4. 让同一个模型在“知道提示”的条件下重新看原来的 response
5. 比较：
   - student：原始上下文下对每个 token 的 logprob
   - teacher：加入 hint 后对同一 response 每个 token 的 logprob
6. 两者之差就是 token-level advantage

核心公式：

\[
A_t[k] = \log \pi_{teacher}(a_t[k] \mid s_{enhanced}) - \log \pi_{\theta}(a_t[k] \mid s_t)
\]

这是什么意思？

- 如果某个 token 在“知道提示后”更合理，teacher 给它更高概率，那么 advantage 为正
- 如果某个 token 在“知道提示后”显得不合理，teacher 给它更低概率，那么 advantage 为负

这就比 Binary RL 精细得多：

- 同一段 response 里，不同 token 可以被不同方向更新

### 3.3.3 为什么这件事从第一性原理上成立？

可以这样理解：

- 原始学生模型在不知道纠错信息时，输出了一串 token
- 现在环境/用户已经事后告诉你“哪里不对”
- 如果把这个事后信息补到原始输入里，再让模型重新评估同一串 token
- 那么这个“补了信息的模型分布”就天然包含纠错方向

这本质上是一种：

- hindsight relabeling
- context-enriched self-distillation
- online self-teaching

它不需要：

- 外部更强 teacher
- 偏好对比数据
- 人工标注的逐 token 标签

它只需要：

- next-state 里存在可提炼的纠错信息

### 3.3.4 OPD 为什么会“稀疏但高质量”？

论文和代码都明确体现了一个策略：

- 不是每个样本都做 OPD
- 只有当 next-state 里确实存在清晰、可操作的 hint 时才接收

这意味着：

- OPD 样本更少
- 但每个样本的信息密度更高

所以作者说它和 Binary RL 是互补的：

- Binary RL：广覆盖，低分辨率
- OPD：低覆盖，高分辨率

---

## 3.4 方法三：Combine

Combine 很自然：

- Binary RL 提供 sequence-level evaluative signal
- OPD 提供 token-level directional signal

于是直接线性相加：

\[
A_t = w_{binary} \cdot r_{final} + w_{opd} \cdot \left(\log \pi_{teacher} - \log \pi_\theta\right)
\]

论文默认：

- `w_binary = 1`
- `w_opd = 1`

它的直觉很朴素：

- “整体上这段好不好” 和 “局部哪些 token 应该往哪边改” 并不冲突

这个方法在论文实验里效果最好。

---

## 3.5 一般 agent 任务中的 Step-wise Reward

### 3.5.1 为什么一般 agent 任务不能只看终局奖励？

因为 terminal / GUI / SWE 这类任务经常很长：

- GUI 30 步
- SWE 20 步
- terminal 10 步

如果只看最终成败：

- 中间绝大多数动作没有直接监督
- credit assignment 会很差

所以作者把每一步的 next-state 也交给 PRM judge：

- 判断这一步有没有推进任务
- 给出 `+1 / -1`

然后把每步 process reward 与终局 outcome reward 相加。

### 3.5.2 论文里一个容易被忽略但很关键的点

对于一般 agent 任务，论文不是说“简单把每步分数扔进去就行”，而是还做了一个**按 step index 分组的标准化**。

原因是：

- 真实环境状态很难聚类
- 但“第 3 步动作”和“第 8 步动作”至少处于相对类似的轨迹位置

所以论文采用：

- 以 `(task group, step_index)` 为桶
- 在桶内对 step-wise reward 标准化

这点在代码里落得非常清楚，后面我会对上。

---

## 4. 仓库结构：你应该把哪些目录视为“论文方法的主体”

我建议把仓库分成 3 层来看。

### 4.1 第一层：论文方法的核心实现

这些目录直接对应论文的训练方法：

- `openclaw-rl/`：个人 agent 的 Binary RL
- `openclaw-opd/`：个人 agent 的 OPD
- `openclaw-combine/`：个人 agent 的混合方法
- `openclaw-tinker/`：把上述方法部署到 Tinker 云训练
- `terminal-rl/`：终端 agent
- `gui-rl/`：GUI agent
- `swe-rl/`：SWE agent
- `toolcall-rl/`：工具调用 agent

### 4.2 第二层：训练基础设施接口层

这些目录不是论文“发明”的，但它们决定了训练数据怎么进入训练框架：

- `slime/`
- `Megatron-LM/`

你不需要把它们当成 OpenClaw-RL 的研究贡献，但必须理解它们在这里扮演的角色：

- `slime`：负责 rollout、数据汇总、样本转训练 batch、异步训练编排
- `Megatron-LM`：负责底层大模型训练，尤其是 RL 训练路径

### 4.3 第三层：上游环境/样例/资源

这些更多是环境和资源，不是这篇论文的方法核心：

- `openclaw/`：个人 assistant 上游项目
- `swe-rl/mini-swe-agent/`
- `gui-rl/evaluation_examples/`
- 大量 provider / evaluator / JSON 样例

这些我也看了它们和主链路的连接点，但不会在这里逐文件展开。

---

## 5. 整个系统的总数据流：先建立一张脑内总图

## 5.1 论文里的四环异步架构

论文第 3 节的主张是：四个环完全解耦。

1. Policy serving
2. Environment hosting
3. Reward judging
4. Policy training

用人话说：

- 模型继续服务用户
- 环境继续执行动作
- judge 在后台评价上一步
- trainer 在后台更新权重
- 这些彼此尽量不阻塞

### 5.2 代码里的对应关系

在代码里，大致对应成：

- 服务侧：
  - `openclaw_api_server.py`
  - `openclaw_opd_api_server.py`
  - `openclaw_combine_api_server.py`
  - `openclaw-tinker/api_server.py`
- rollout 收集：
  - `openclaw_rollout.py`
  - `openclaw_opd_rollout.py`
  - `openclaw_combine_rollout.py`
  - `openclaw-tinker/rollout.py`
- 奖励/教师评估：
  - `openclaw_api_server.py` 内的 PRM 逻辑
  - `openclaw_opd_api_server.py` 内的 hint judge + teacher logprob
  - `openclaw-tinker/scorers.py`
  - `terminal-rl/agent/prm_agent.py`
  - `gui-rl/agents/qwen3vl_reward_agent.py`
  - `swe-rl/swe_prm.py`
  - `toolcall-rl/generate_with_retool.py` 内 PRM
- 训练侧：
  - `slime/slime/ray/rollout.py`
  - `slime/slime/backends/megatron_utils/loss.py`
  - `openclaw-combine/combine_loss.py`
  - `openclaw-opd/topk_distillation_loss.py`
  - `openclaw-tinker/trainer.py`

### 5.3 个人 agent 路径的一条完整样本链路

最典型的一条链是这样的：

1. 当前请求到来
2. server 先把这一轮 response 的 token / logprob 记下来
3. 暂时不立刻训练，因为还没有 `s_{t+1}`
4. 等下一轮 main-line turn 到来
5. 从新请求里抽出上一轮的 next-state
6. PRM 或 OPD judge 异步评估上一轮
7. 一旦结果准备好，把 `Sample` 提交到训练队列
8. rollout worker 收集到一批样本
9. `slime` / `Megatron` 开始训练
10. 新权重切换到 serving 端继续服务

注意这个设计非常关键：

> 训练样本不是“当前轮立即产生”，而是“上一轮在下一轮到来时才最终闭环”。

这就是为什么 next-state 在代码中天然表现成“下一次消息的最后一条内容”。

---

## 6. 核心数据结构：论文概念在代码里是怎么被装起来的

## 6.1 `Sample` 是整个系统的中枢

文件：

- `slime/slime/utils/types.py`

`Sample` 基本可以看成 OpenClaw-RL 和 `slime` 之间的通用样本容器。

关键字段：

- `prompt`
- `tokens`
- `response`
- `response_length`
- `reward`
- `loss_mask`
- `rollout_log_probs`
- `metadata`
- `train_metadata`

你可以把它理解为：

- `prompt/tokens/response`：模型到底看了什么、说了什么
- `reward`：最终怎么奖励
- `loss_mask`：哪些 response token 真的参与训练
- `rollout_log_probs`：rollout 时旧策略的 logprob，用于 PPO/GRPO 风格 loss
- `metadata`：额外的过程监督信息，比如 step-wise reward

### 6.2 `loss_mask` 为什么非常关键？

很多人第一次看这类代码，会把 `response` 当成“整段都训练”。
但 agent 场景不是这样。

因为 response 里往往混着：

- 模型自己生成的 assistant token
- 环境返回的 observation
- 工具执行结果
- 模板 token

真正应该训练的，只有模型自己生成的那部分 token。

所以代码大量在做一件事：

- 给 response 中每个 token 打 `0/1`
- `1` 表示这个 token 参与 policy gradient
- `0` 表示这个 token 是环境/模板/观测，不训练

这也是为什么：

- terminal
- GUI
- SWE
- tool-call

这些子项目都要非常认真地重建 token span 和 loss mask。

### 6.3 `metadata["step_wise"]` 是一般 agent 任务的桥梁

一般 agent 任务不是只存一个 reward，而是额外存：

- 每一步对应哪个 token span
- 每一步 step reward 是多少
- step reward 是否已与 outcome reward 合成

代码里常见结构是：

- `metadata["step_wise"]["step_scores"]`
- `metadata["step_wise"]["step_scores_with_outcome"]`
- `metadata["step_wise"]["step_token_spans"]`

这个结构后面会被 `slime` 读取，再广播到 token 级别 advantage。

---

## 7. 个人 agent 轨道：Binary RL / OPD / Combine 的代码级解剖

## 7.1 Binary RL：`openclaw-rl/`

核心文件：

- `openclaw-rl/openclaw_api_server.py`
- `openclaw-rl/openclaw_rollout.py`
- `openclaw-rl/run_qwen3_4b_openclaw_rl.sh`

### 7.1.1 `openclaw_api_server.py` 在做什么？

这个文件本质上是一个 OpenAI-compatible API 代理，但它不是普通代理。

它额外做了五件大事：

1. 把会话按 session/turn 管起来
2. 区分 main-line turn 和 side turn
3. 缓存上一轮的 response/logprob
4. 等下一轮消息到来后，抽出上一轮的 next-state
5. 异步触发 PRM 评分并提交训练样本

这非常符合论文第 3.2 节的“session-aware environment server”。

### 7.1.2 main-line 和 side turn 的意义

论文里说：

- main-line turn 是主要可训练交互
- side turn 是辅助查询、记忆整理、环境过渡

代码里也是这个思路：

- 只有 `turn_type == "main"` 的交互才会成为训练候选
- side turn 会转发，但不产训练数据

这背后的第一性原理是：

- 不是所有 API 请求都代表“用户对主策略的一次真实反馈”
- 如果不做筛选，训练会被很多杂质请求污染

### 7.1.3 PRM 评分的关键实现点

`openclaw_api_server.py` 里有几组关键辅助函数：

- `_build_prm_judge_prompt`
- `_parse_prm_score`
- `_majority_vote`

逻辑很直接：

1. 用上一轮 response + 当前 next-state 组成 judge prompt
2. 多次并发调用 judge
3. 解析 boxed score
4. 多数投票

这里的工程处理很稳健：

- 不可解析输出会落成 `0`
- 多次投票降低 PRM 抖动

### 7.1.4 最关键的数据闭环：上一轮要等下一轮才可训练

代码里有一个很重要的时序：

- 当前 main turn 先生成 response
- 只缓存，不立刻训练
- 等下一次 main turn 到来时
- 取新消息里的最后内容当作旧 turn 的 `next_state`
- 再对旧 turn 发起 PRM

这正是论文里 Binary RL 的 Algorithm 1。

### 7.1.5 一个论文里没强调、代码里很重要的工程细节：at-least-one guarantee

这个细节很值得你记住。

在 `openclaw_api_server.py` 中，如果：

- 一个 session 里还没有任何有效训练样本
- 当前 PRM 结果又是 `0`

代码会触发一个“至少保留一个有效样本”的保证，强制让样本可训练。

这不是论文主公式里的重点，但它是非常典型的在线系统工程补丁。

为什么需要它？

- 真实在线对话里，很多轮 next-state 可能比较模糊
- 如果一整个 session 都被过滤掉，训练吞吐会过低

所以代码宁愿保底留一个样本，也不希望整个 session 毫无训练贡献。

### 7.1.6 最后一轮为什么经常不训练？

因为最后一轮没有 `s_{t+1}`。

这在论文里是定义层面的事实，在代码里则表现为：

- 如果一个 turn 没有 next-state
- 通常就会 `loss_mask = 0` 或直接不提交

这再次提醒你：

> OpenClaw-RL 的监督不是凭空来的，它必须来自下一状态。

### 7.1.7 `openclaw_rollout.py` 的作用

`openclaw_rollout.py` 负责：

- 持续从 API server 的输出队列收集 ready samples
- 凑够 `rollout_batch_size`
- 在权重更新时暂停提交
- 在新策略版本开始时清理 record 文件

这个文件体现了论文第 3.5 节“non-blocking record and observability”的一个重要工程选择：

- 日志和训练样本是持续流动的
- 但每次权重更新边界都会清理记录文件，以保证日志对应单一 policy version

### 7.1.8 训练脚本反映了哪些实际超参数？

`run_qwen3_4b_openclaw_rl.sh` 给出的默认设置很重要：

- `rollout-batch-size = 16`
- `rollout-temperature = 0.6`
- `advantage-estimator = grpo`
- `disable-rewards-normalization`
- `eps-clip = 0.2`
- `eps-clip-high = 0.28`
- `kl-loss-coef = 0.0`

这里有个要特别注意的点：

- 论文公式里给了 `β_KL = 0.02`
- 但脚本默认 `kl-loss-coef = 0.0`

这说明：

- 论文给的是方法描述/默认理论形式
- 实际开源脚本在 personal track 上更激进地把 KL 关掉了

这是一个**论文与实际脚本存在偏差**的地方。

---

## 7.2 OPD：`openclaw-opd/`

核心文件：

- `openclaw-opd/openclaw_opd_api_server.py`
- `openclaw-opd/openclaw_opd_rollout.py`
- `openclaw-opd/topk_distillation_loss.py`
- `openclaw-opd/run_qwen3_4b_openclaw_opd*.sh`

### 7.2.1 OPD 的代码主线完全对应论文 Algorithm 2

`openclaw_opd_api_server.py` 基本就是论文 Algorithm 2 的工程化版本。

关键步骤对应如下：

论文步骤：

1. Judge `(a_t, s_{t+1})`
2. 保留正向且 hint 足够长的结果
3. 选最优 hint
4. 拼进原 prompt
5. 强制计算 teacher logprob
6. 提交样本

代码对应函数：

- `_build_hint_judge_messages`
- `_parse_judge_result`
- `_select_best_hint`
- `_append_hint_to_messages`
- `_compute_teacher_log_probs`
- `_opd_evaluate`

### 7.2.2 OPD 最核心的难点不是 hint，而是“对齐”

很多人会以为 OPD 的重点是抽 hint。
其实更难的是：

- 如何在“增强过的 prompt”下
- 对“原来的 response token 序列”
- 精确计算对应的 teacher logprob

这要求：

- token 序列对齐不能错
- response span 不能错
- logprob 长度必须和原 response token 对齐

代码里这部分处理得很认真。

### 7.2.3 为什么 OPD 会直接丢弃很多样本？

`_opd_evaluate` 的逻辑是：

- judge 先判断有没有有效 hindsight
- 只有接受的样本才提交训练
- 没有足够好的 hint 就直接丢弃

这和论文完全一致：

- OPD 故意牺牲样本量换信号质量

这也是为什么论文实验里 OPD 生效更晚：

- 因为样本更稀疏

### 7.2.4 OPD 路径中的 `reward={"score":1.0}` 不要误读

在 OPD 样本里，代码通常把 reward 写成一个看似平凡的值，例如 `1.0`。

但真正关键的不是这个 scalar reward，而是：

- `teacher_log_probs`

也就是说，OPD 的训练信息主要不在 reward，而在 token-level teacher signal 里。

### 7.2.5 `openclaw_opd_rollout.py` 为什么比 RL 更“严格”？

因为 RL 路径几乎所有可评分样本都能进入训练；
但 OPD 路径只收通过质量过滤的样本。

所以 rollout worker 只会排出有效 OPD 样本，不会像 Binary RL 那样尽量保留所有评分样本。

### 7.2.6 top-k distillation 是什么？

`topk_distillation_loss.py` 是一个额外扩展。

它不是论文主线里的必要组件，而是把 OPD 从：

- “只用 teacher 对真实 response token 的 logprob”

扩展成：

- “用 teacher 的 top-k 分布做蒸馏”

这个 loss：

- 读取 `teacher_topk_log_probs`
- 读取 `teacher_topk_indices`
- 再给剩余 vocabulary 质量加一个 tail bin
- 计算 reverse KL

它引用了 SDFT / SDPO 的思路。

这说明作者把 OPD 不只是当论文方法，也当一个可以继续接蒸馏族方法的接口层。

---

## 7.3 Combine：`openclaw-combine/`

核心文件：

- `openclaw-combine/openclaw_combine_api_server.py`
- `openclaw-combine/openclaw_combine_rollout.py`
- `openclaw-combine/combine_loss.py`
- `openclaw-combine/run_qwen3_4b_openclaw_combine*.sh`

### 7.3.1 Combine 在代码里不是“同时做两份训练”，而是“合成一种 advantage”

这一点很重要。

代码不是分两套 loss：

- 一份 Binary RL loss
- 一份 OPD loss

然后再加权。

而是：

- 先构造 combined advantage
- 再走同一个 PPO-style policy loss

这和论文表述一致。

### 7.3.2 `openclaw_combine_api_server.py` 的样本分流逻辑

代码把每个 turn 的情况分成四类：

1. hint 有、eval 有：`opd+rl`
2. hint 有、eval 中性：`opd`
3. hint 没有、eval 有：`rl`
4. 两者都没有：丢弃

这非常漂亮，因为它把论文里的“互补性”做成了显式分支。

### 7.3.3 一个非常巧妙的工程技巧：RL-only 样本如何让 OPD 分量自然为 0

在 RL-only 样本里，代码会设置：

- `teacher_log_probs = rollout_log_probs`

这样一来：

\[
\log \pi_{teacher} - \log \pi_{old} \approx 0
\]

于是 combined advantage 自动只剩 RL 分量。

这个技巧很妙，因为它避免了：

- 为不同样本类型写完全不同的 loss 分支

### 7.3.4 `combine_loss.py` 是整个 combine 方法的数学核心

这个文件直接把 batch 中的两部分优势取出来：

- `grpo_advantages`
- `teacher_advantages = teacher_log_probs - old_log_probs`

然后：

\[
combined = w_{opd} \cdot teacher\_advantages + w_{rl} \cdot grpo\_advantages
\]

再送入标准 PPO clipped objective。

这里的几个重要点：

- Combine 没有发明新 optimizer
- 它只是重写了 advantage 组合方式
- loss 仍然是 PPO 风格

### 7.3.5 Combine 脚本里的实际设置

`run_qwen3_4b_openclaw_combine.sh` 里：

- `OPENCLAW_COMBINE_W_RL=1.0`
- `OPENCLAW_COMBINE_W_OPD=1.0`
- `loss-type = custom_loss`
- `custom-loss-function-path = combine_loss.combine_loss_function`

这说明 combine 方案不是训练框架原生自带，而是通过自定义 loss 注入 `slime` / `Megatron`。

---

## 7.4 Tinker 版本：`openclaw-tinker/`

这是我认为仓库里非常值得读的一部分，因为它把论文方法从“本地 SGLang + Megatron 方案”推广到了“云端 Tinker 方案”。

核心文件：

- `openclaw-tinker/config.py`
- `openclaw-tinker/run.py`
- `openclaw-tinker/api_server.py`
- `openclaw-tinker/scorers.py`
- `openclaw-tinker/data_formatter.py`
- `openclaw-tinker/rollout.py`
- `openclaw-tinker/trainer.py`

### 7.4.1 它本质上在做什么？

`openclaw-tinker/` 的核心不是改变算法，而是改变执行底座：

- 不再直接依赖本地 SGLang / Megatron 训练编排
- 而是把 policy、teacher、judge 都放到 Tinker 服务上

这意味着：

- 训练逻辑仍然是 RL / OPD / Combine
- 但 serving / sampling / weight save / lora swap 都通过 Tinker client 完成

### 7.4.2 `trainer.py` 是最重要的入口

`trainer.py` 的训练循环非常清楚：

1. 收集 batch
2. 把 batch 转成 Tinker datums
3. 前反向
4. optimizer step
5. 保存权重并获得新的 sampling client
6. 更新 rollout server 的采样客户端

一个非常关键的工程点是：

- **只在 weight swap 时短暂停止 submission**
- 不是整个训练阶段都停服

这正对应论文的 fully async 精神。

### 7.4.3 `data_formatter.py` 很值得仔细读

这个文件把 OpenClaw 自己的 `TrainingSample` 转成 Tinker 需要的 `Datum`。

它清楚体现了三种方法的区别：

- RL / OPD：
  - 用 scalar reward 或其广播
  - 可选叠加 reverse-KL penalty
- Combine：
  - 每个 token 上都构造
    - `w_rl * reward`
    - `w_opd * (-kl_coef * (student - teacher))`

这里再次说明：

> “Combine”在工程上真正实现为“每个 token 的 advantage 组合”，而不是抽象口号。

### 7.4.4 Tinker 版本与本地版本的一个重要差异

本地 `slime` 路径里，GRPO/step-wise 等 advantage 逻辑主要在训练框架侧。

但在 Tinker 里，`data_formatter.py` 更直接地把 advantage 写进 datum。

也就是说：

- 本地路径：更多依赖 `slime/Megatron` 的 batch 后处理
- Tinker 路径：更多在 OpenClaw 自己这一层先做好 token-level advantages

### 7.4.5 `api_server.py` 的一个关键工程点：严格保留原始 token/logprob 对齐

Tinker 版本的 `api_server.py` 特别强调保留：

- raw prompt ids
- raw response tokens
- raw response logprobs

这是因为：

- OPD / Combine 都对 token 对齐极其敏感
- 一旦中间被模板化重编码，teacher/student 对齐就容易错

所以这个文件的价值不只是“代理请求”，而是“保证蒸馏和 PPO 的 token 对齐不被破坏”。

---

## 8. 一般 agent 轨道：terminal / GUI / SWE / tool-call 如何共享同一逻辑

如果把个人 agent 轨道的思路抽象出来，一般 agent 轨道其实就变成：

1. 生成一步 action
2. 在真实环境中执行
3. 拿到 next observation / tool result / test result
4. 让 PRM 判断这一步有没有推进任务
5. 把 step reward 和记为当前动作对应 token span 的监督
6. 最后再加 outcome reward

所以这几个子目录虽然环境不同，但训练逻辑很一致。

---

## 8.1 Terminal Agent：`terminal-rl/`

核心文件：

- `terminal-rl/generate.py`
- `terminal-rl/agent_runner.py`
- `terminal-rl/agent/camel_agent.py`
- `terminal-rl/agent/prm_agent.py`
- `terminal-rl/inference_client.py`
- `terminal-rl/router_server.py`
- `terminal-rl/remote/pool_server.py`
- `terminal-rl/remote/terminal_env.py`
- `terminal-rl/env_client.py`

### 8.1.1 它在环境层面是什么？

terminal agent 的环境不是一个抽象 simulator，而是一个真实 shell execution sandbox。

论文表 1 里说 terminal 的 next-state 是：

- stdout
- stderr
- exit code

代码里这件事被具体化成：

- `TerminalEnvClient.exec_tool()`
- `terminal_env.py` 中的 shell / file viewer 等工具

### 8.1.2 `generate.py` 是主循环

它做的事情可以概括成：

1. 申请一个远程 terminal 环境 lease
2. reset 到某个任务
3. 用 rollout policy 产生一轮带 tool-call 的 response
4. 执行工具
5. 得到 observation
6. 如果启用 PRM，则异步评价这一步
7. 重复直到结束
8. 最后用环境 evaluator 给 outcome reward

### 8.1.3 step-wise reward 在 terminal 路径里如何落地？

`generate.py` 的 reward 逻辑大致是：

- outcome reward：`2 * accuracy - 1`
- 如果有 PRM step 分数，就把它加进去
- 同时把 step-wise 信息写到 `sample.metadata["step_wise"]`

也就是说，论文中的：

- “process reward + outcome reward”

在 terminal 代码里是明确的、可追踪的字段，而不是隐藏在黑箱里。

### 8.1.4 `camel_agent.py` 很值得注意的一点

它不是只负责生成，还负责：

- 解析工具调用
- 在工具调用格式错误时构造 synthetic tool error turn

这其实很符合论文观点：

- 工具调用失败本身也是 next-state
- 失败不该被丢弃，而应该回流进训练

### 8.1.5 terminal 环境为什么要有 router + pool 两层？

代码里有两层环境服务：

- `router_server.py`
- `remote/pool_server.py`

粗略理解：

- router：把任务 sticky 地路由到某个 worker
- pool server：在 worker 上管理很多运行中的 terminal 环境 lease

这说明作者不只是做算法，而是在认真解决：

- 环境并发
- 长任务资源占用
- lease 生命周期

这也是论文第 3.3 节“从单用户到大规模环境并行”的基础。

---

## 8.2 GUI Agent：`gui-rl/`

核心文件：

- `gui-rl/generate_with_gui.py`
- `gui-rl/agents/qwen3vl_agent.py`
- `gui-rl/agents/qwen3vl_reward_agent.py`
- `gui-rl/env_pool_server.py`
- `gui-rl/desktop_env/desktop_env.py`

### 8.2.1 GUI 场景为什么比 terminal 更难？

因为 terminal 主要是文本状态转移，GUI 则是：

- 截图
- 鼠标键盘动作
- 屏幕可视变化

这里 next-state 不是简单的 stdout，而是：

- 当前截图
- 执行动作后的下一张截图
- 可能还包括 accessibility / evaluator 信息

### 8.2.2 `qwen3vl_agent.py` 负责什么？

这个文件是 policy 侧多模态 agent 的核心。

它负责：

- 构造 GUI tool spec
- 组织系统提示
- 处理截图缩放
- 构造多模态 prompt
- 解析模型输出的 `<tool_call>` JSON
- 转成 `pyautogui` 风格动作
- 生成训练用 `loss_mask`

这里有两个很关键的点：

1. GUI action 不是自由文本，而是被约束成 tool call
2. 多模态训练同样要严肃地构造 token-level loss mask

### 8.2.3 `qwen3vl_reward_agent.py` 是 GUI 版 PRM

它会把下面这些东西一起喂给 judge：

- 之前的动作历史
- 当前截图
- agent 当前 response
- 执行动作后的下一张截图

然后判断：

- 这一步是否真的让界面朝目标推进

这正是论文里“use the next observation AFTER executing this step to judge whether the action actually took effect”的具体实现。

### 8.2.4 GUI 的 reward 组合

在 `generate_with_gui.py` 中：

- outcome reward 仍然来自最终任务是否完成
- PRM 则给每一步打分
- reward_func 把二者组合起来
- 再把结果写进 `metadata["step_wise"]`

还支持一种 `dynamic_history` 模式：

- 每步都可以单独形成样本

这个模式对长轨迹任务很重要，因为它能减少超长上下文带来的训练难度。

### 8.2.5 `env_pool_server.py` 和 `desktop_env.py`

这两者体现了 GUI 环境的工程成本。

- `env_pool_server.py`：负责复用桌面环境实例
- `desktop_env.py`：负责 reset、snapshot、step、evaluate

这也是为什么 GUI PRM 比文本 PRM 更贵：

- 不只是模型更贵
- 环境本身也更贵

---

## 8.3 SWE Agent：`swe-rl/`

核心文件：

- `swe-rl/generate_with_swe_remote.py`
- `swe-rl/swe_prm.py`
- `swe-rl/message_utils.py`
- `swe-rl/swe_env_client.py`
- `swe-rl/server/swe_env_pool_server.py`

### 8.3.1 SWE 环境本质上是什么？

SWE 场景不是一般命令行任务，而是：

- 一个真实代码仓库
- 一组待修 issue
- 一个测试/eval 流程

next-state 包括：

- bash 命令输出
- diff
- 测试结果
- 最终 patch 是否 resolve issue

### 8.3.2 `generate_with_swe_remote.py` 的核心流程

它大体做这些事：

1. 申请远程 docker 环境
2. 用多轮 prompt 驱动 agent 输出 bash 命令
3. 在容器内执行命令
4. 把命令结果作为下一状态
5. 如启用 PRM，则评价当前步
6. 直到 agent 提交 patch
7. 再在新评测容器里应用 patch 并运行 eval script
8. 把是否 resolved 作为 outcome reward

### 8.3.3 `swe_prm.py` 把论文里的 SWE judge prompt 写成了代码

它使用的信息包括：

- issue description
- recent history
- 当前 step 的完整 response
- 当前 bash 命令执行结果

然后做 m-voting，输出 step-level `+1/-1`。

这与论文附录 prompt 模板一致性很高。

### 8.3.4 `message_utils.py` 非常关键，但容易被忽略

SWE 轨迹经常是多轮聊天 + bash 结果混合。

为了训练，你必须准确知道：

- 哪些 token 是 assistant 真正生成的
- 哪些只是模板和环境回放

`message_utils.py` 正在做这件事：

- assistant generation prompt：mask 0
- assistant 真正生成内容：mask 1
- trailing template token：mask 0

如果这一步对齐错了，训练就会把环境文本当成模型行为去优化，后果很严重。

### 8.3.5 SWE 的环境并发实现

`swe_env_pool_server.py` 管理远程 ECS docker 节点：

- allocate lease
- exec
- diff
- evaluate
- close

这让作者能把 SWE 训练从“本地单容器”扩展到“远程多容器并发”。

---

## 8.4 Tool-call Agent：`toolcall-rl/`

核心文件：

- `toolcall-rl/generate_with_retool.py`
- `toolcall-rl/tool_sandbox.py`
- `toolcall-rl/rl_data_preprocess.py`
- `toolcall-rl/sft_data_processing.py`

### 8.4.1 这是整个仓库里最“教科书式”展示 next-state 思想的子项目

因为这个环境非常纯粹：

- 模型输出代码
- 沙箱执行
- 返回解释器输出

这就是最标准的：

- 动作：写代码/给答案
- next-state：代码执行结果

所以它特别适合拿来理解论文方法。

### 8.4.2 `generate_with_retool.py` 的主循环

逻辑是：

1. 把问题包装成带工具定义的对话模板
2. 调模型生成
3. 如果是 `<tool_call>`，则执行 Python 代码
4. 把 `<interpreter>` 输出拼回上下文
5. 继续生成
6. 如果是 `Answer: \boxed{...}`，则结束

每轮动作都会记录：

- action token span
- logprob
- 是否调用工具
- PRM step score

### 8.4.3 它也支持 step-wise PRM

`generate_with_retool.py` 在每一步工具执行后都可以做 PRM judge：

- 用问题
- 历史轨迹
- 当前动作
- 当前 observation

构造一个 step-wise PRM prompt。

最终：

- `sample.metadata["prm"]` 保存每步 judge 细节
- `sample.metadata["step_wise"]` 保存 token span 与 step score 对齐信息

### 8.4.4 reward 组合里还有一个有趣的细节

它的 outcome reward 是 math_dapo 分数。

此外，如果答案错了，代码还会给“调用工具本身”一点正向鼓励，避免模型在错误时完全不愿探索工具。

这说明 tool-call 项目不只是照搬论文，而是做了具体任务上的 reward shaping。

### 8.4.5 `tool_sandbox.py` 是环境安全边界

这个文件定义了：

- 模块白名单
- 危险模式检测
- timeout
- memory limit
- 最大并发

它限制：

- `os`
- `sys`
- `subprocess`
- `eval`
- `exec`
- `open`
- 双下划线方法等

这说明 tool-call agent 的环境不是“任意执行代码”，而是一个严格收缩过的 Python 解释器任务环境。

---

## 9. `slime` / `Megatron-LM` 在这里到底扮演什么角色

这是理解整个仓库最容易混乱的地方。

## 9.1 `slime` 不是论文方法本身，但它决定训练样本怎么流动

最重要的几个接口层文件是：

- `slime/slime/rollout/sglang_rollout.py`
- `slime/slime/ray/rollout.py`
- `slime/slime/backends/megatron_utils/loss.py`

### 9.1.1 `sglang_rollout.py`

这里定义了：

- `GenerateState`
- 通用 `generate()`
- `Sample` 的填充方式

OpenClaw-RL 的很多子项目都通过：

- `--custom-generate-function-path`
- `--custom-rm-path`

把自己的生成逻辑和 reward 逻辑插进 `slime`。

所以 `slime` 提供的是：

- 统一训练框架
- 统一样本容器
- 统一 rollout 调度

OpenClaw-RL 提供的是：

- 不同场景下如何生成样本
- 如何定义 reward
- 如何定义 custom loss

### 9.1.2 `slime/slime/ray/rollout.py`：step-wise reward 的关键桥梁

这个文件里最重要的是 `_post_process_step_wise_rewards()`。

它会读取每个样本的：

- `step_scores_with_outcome`
- `step_token_spans`
- `step_indices`

然后：

1. 对齐长度
2. 按 `(group_index, step_index)` 统计
3. 如果开启 reward normalization，就在桶内标准化
4. 如果某个桶内完全常数，可能丢弃
5. 最终写入 `train_data`

这几乎就是论文第 4.4.2 节的工程实现。

### 9.1.3 `slime/slime/backends/megatron_utils/loss.py`

这里的 `compute_advantages_and_returns()` 很关键。

对于 `advantage_estimator == "step_wise"`：

- 它不会重新发明 reward
- 而是把每步标量 reward 广播到对应 token span

也就是说：

- 先有 step-level reward
- 再根据 `token_start/token_end` 扩展成 token-level advantage

这是理解“step-wise reward 最终怎么训练到 token 上”的关键代码。

## 9.2 `Megatron-LM` 是底层训练引擎

在这个仓库里，`Megatron-LM` 不是研究重点，但它承担：

- 大模型并行训练
- RL 训练的底层张量与损失计算

个人 track 的 `combine_loss.py` 和 OPD 扩展 loss，都是在它和 `slime` 约定的接口上插进去的。

所以可以这样记：

- 论文方法在 OpenClaw-RL 目录里
- 大规模训练骨架在 `slime` / `Megatron-LM`

---

## 10. 论文与代码的“对得上”和“对不上”

这一节非常重要，因为很多项目论文和代码并不完全一致。

## 10.1 高度一致的地方

### 10.1.1 next-state 是统一反馈源

论文说：

- user reply
- stdout/stderr
- GUI state change
- test verdicts
- tool returns

代码完全对应：

- personal agent：下一轮 main-line message
- terminal：tool execution result
- GUI：next screenshot
- SWE：command output / tests / diff
- tool-call：interpreter output

### 10.1.2 Binary RL 的多数投票 judge

论文讲 m-vote majority vote；
代码在 personal、GUI、SWE、tool-call 路径里都做了对应实现。

### 10.1.3 OPD 的四步式处理

论文：

- 判断是否有 hint
- 选最长有效 hint
- 拼回 prompt
- 算 teacher-student logprob gap

代码严格照此实现。

### 10.1.4 Step-wise reward 的“按 step index 标准化”

论文：

- 现实状态难聚类
- 按 step index 分组更务实

代码：

- `slime/slime/ray/rollout.py` 中按 `(group_index, step_index)` 标准化

这一点非常一致。

## 10.2 论文没强调，但代码里很关键的工程补丁

### 10.2.1 at-least-one guarantee

前面讲过：

- 如果一个 session 一直没有效样本，代码会尽量保底留下一个

这是在线系统为提升训练吞吐做的工程兜底。

### 10.2.2 最后一轮没有 next-state 时通常不训练

论文里这是隐含事实，代码里则是显式分支。

### 10.2.3 日志按权重版本切分

论文 3.5 提到 non-blocking record；
代码里会在权重更新边界 purge record file，避免不同 policy version 混在一起。

### 10.2.4 Combine 的“RL-only 样本 teacher=student”技巧

论文只给了组合公式；
代码给出了一个非常具体且优雅的实现技巧。

### 10.2.5 Tinker 版本只在权重切换时暂停

这比“训练时全停服”更符合 async 系统目标。

## 10.3 论文与脚本配置之间的偏差

### 10.3.1 KL 系数并不完全一致

论文方法部分写了类似：

- `β_KL = 0.02`

但开源脚本里：

- personal RL / combine 常见是 `kl-loss-coef = 0.0`
- general agent 路径则更常见 `0.01`

所以别把论文里的公式默认值直接当成所有脚本的实际运行值。

### 10.3.2 PRM 投票数也因场景而异

论文表 5 说：

- GUI 是 3
- 其他常见是 1

代码脚本也基本体现了这种差异。

这再次说明：

- 不同环境下 judge 成本和收益不同

## 10.4 一些需要保持审慎的地方

### 10.4.1 OPD 依赖 hint 抽取质量

如果 judge 抽出来的 hint 不够聚焦，OPD 可能把 noisy hindsight 强行注入 teacher context。

### 10.4.2 Binary RL 的 reward 过于粗粒度

虽然覆盖广，但在长而复杂的 response 上，sequence-level scalar 的信用分配仍然很粗。

### 10.4.3 step-wise PRM 成本不低

论文也承认：

- 引入 PRM 往往会提升效果
- 但资源开销会上升

GUI 和 SWE 里尤其明显，因为环境和 judge 都贵。

---

## 11. 实验结果应该怎么理解，而不是只背表格

## 11.1 Personal Track：为什么 Combined 最好？

论文表 3：

- base score = 0.17
- 8 steps 更新后：
  - Binary RL = 0.25
  - OPD = 0.25
  - Combined = 0.76
- 16 steps 更新后：
  - Binary RL = 0.23
  - OPD = 0.72
  - Combined = 0.81

### 11.1.1 这组结果说明了什么？

第一，Combined 确实不是口头互补，而是实验上强互补。

第二，OPD 的提升往往更晚显现。

这和代码完全一致，因为：

- OPD 样本会严格过滤
- 起步阶段进入训练的数据少

第三，Binary RL 单独使用的上限似乎有限。

原因也很直观：

- 它提供的是粗粒度序列监督
- 能把风格大方向推一推
- 但不擅长给出细致风格修正

## 11.2 General Track：为什么 process reward 会带来增益？

论文表 4：

- Tool-call：
  - integrated = 0.30
  - outcome only = 0.17
- GUI：
  - integrated = 0.33
  - outcome only = 0.31

### 11.2.1 应该如何解读这组结果？

我会比较谨慎地解读：

- 它说明 step-wise process reward 在长时程任务里确实有效
- 但不同场景的增益幅度不一样

tool-call 提升更明显，因为：

- 工具执行结果非常直接
- 下一状态和动作质量的对应关系比较清楚

GUI 提升较小但仍为正，因为：

- GUI 的下一状态更复杂、更难判
- PRM judge 本身也更难

### 11.2.2 这不是“PRM 一定神奇”，而是“长时程信用分配需要中间信号”

真正该记住的不是表格数字本身，而是：

> 长时程 agent 的难点不是最后一步答对，而是中间 20 多步里大量动作都缺乏明确监督；PRM 的价值在于给这些中间步补上信用分配。

---

## 12. 我认为这篇工作真正有价值的地方

## 12.1 它抓住了一个非常强的统一抽象

很多 agent RL 工作只做单一环境：

- 只做 SWE
- 只做 GUI
- 只做 math tool use

OpenClaw-RL 真正强的地方是：

- 用 next-state 做统一抽象
- 于是聊天、终端、GUI、SWE、tool-call 都变成同一类问题

## 12.2 它把“在线持续训练”认真做成系统

很多论文会说 continual / online / live training，但代码实际上仍偏 batch。

这个仓库至少在核心路径上是认真做了：

- async serving
- async judging
- async training
- weight swap
- record versioning

也就是说，它不是只在算法上说得漂亮，系统层也下了功夫。

## 12.3 OPD 的设计很有启发性

我认为这篇论文最有思想含量的部分不是 Binary RL，而是 OPD。

因为它抓住了一件很重要的事：

- 真实世界反馈经常不是纯标量
- 很多反馈其实天然带有“怎么改”的信息

把这类信息重新编码成 token-level directional advantage，是比“只打一个分数”更接近人类纠错方式的。

---

## 13. 我认为这篇工作也有哪些边界和限制

## 13.1 它很依赖 judge / PRM 的质量

无论是：

- Binary RL
- OPD
- step-wise process reward

本质都依赖 judge 把 next-state 正确解释出来。

如果 judge 不稳定：

- Binary RL 会给错粗奖励
- OPD 会抽错 hint
- step-wise reward 会把错误信用分配广播到整段 token

## 13.2 “统一框架”不代表“所有环境同样容易”

next-state 这个抽象是统一的，但不同环境难度差很多：

- tool-call：最干净
- terminal：中等
- SWE：复杂
- GUI：最脏、最贵、最难判

所以不要把“统一框架”理解成“所有场景都一样简单”。

## 13.3 个人 agent 结果更偏演示性质

personal track 的实验很有趣，也有说服力，但它更像 proof-of-concept：

- 数据量不大
- 偏好模拟成分较强
- 任务集中在特定人设

它证明了方法可行，但还不是那种“大规模、严格 benchmark 化”的 personalization 定论。

---

## 14. 如果你要自己继续深入，我建议的阅读顺序

### 第一轮：只抓主线

按这个顺序：

1. `README.md`
2. 论文第 2、3、4、5 节
3. `openclaw-rl/openclaw_api_server.py`
4. `openclaw-opd/openclaw_opd_api_server.py`
5. `openclaw-combine/combine_loss.py`
6. `slime/slime/ray/rollout.py`
7. `slime/slime/backends/megatron_utils/loss.py`

### 第二轮：看一般 agent 场景

1. `terminal-rl/generate.py`
2. `gui-rl/generate_with_gui.py`
3. `swe-rl/generate_with_swe_remote.py`
4. `toolcall-rl/generate_with_retool.py`

### 第三轮：看部署与替代底座

1. `openclaw-tinker/trainer.py`
2. `openclaw-tinker/api_server.py`
3. `openclaw-tinker/data_formatter.py`

---

## 15. 最后的总结：把整件事重新压缩成三层理解

### 第一层：一句最本质的话

**OpenClaw-RL 认为，agent 每一步之后得到的 next-state，本身就是最自然、最通用的在线学习信号。**

### 第二层：三种训练信号恢复方式

1. next-state 只表达“好/坏”
   - 用 Binary RL
2. next-state 还表达“应该怎么改”
   - 用 OPD
3. 任务很长，最终奖励太稀疏
   - 用 step-wise PRM + outcome reward

### 第三层：代码层面的真实落地

- personal track：
  - 上一轮 response 先缓存
  - 等下一轮拿到 next-state
  - 再触发 PRM / OPD
- general track：
  - 每执行一步动作
  - 马上得到 next observation
  - 立即异步打 step reward
- 训练框架：
  - `Sample` 承载数据
  - `metadata["step_wise"]` 承载过程监督
  - `slime` / `Megatron` 把它们变成 token-level advantage

如果你把这三层都理解了，这个项目基本就真的吃透了。

---

## 16. 一页版极简结论

- 这篇论文最强的点，不是某个新 loss，而是把“next-state 作为统一反馈源”这件事讲清楚并做成了系统。
- Binary RL 解决“有评价、没方向”的问题。
- OPD 解决“有方向、别浪费”的问题。
- Combine 之所以有效，是因为它把粗粒度和细粒度监督真正合并到了同一 advantage 里。
- 一般 agent 场景真正重要的是 step-wise credit assignment，代码里靠 `step_token_spans + step_scores_with_outcome` 实现。
- 仓库最值得精读的不是所有上游代码，而是：
  - `openclaw-rl/`
  - `openclaw-opd/`
  - `openclaw-combine/`
  - `openclaw-tinker/`
  - 以及 `slime` 中 step-wise / advantage 那几处接口层实现。

