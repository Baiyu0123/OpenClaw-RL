# 实验操作手册

## 快速参考

```bash
# 停止 → 清理 → 启动新实验（完整流程，约 2 分钟）
cd ~/OpenClaw-RL
bash openclaw-combine/stop_combine.sh
bash experiment/cleanup.sh exp27        # 归档旧数据 + 清理 checkpoint
bash openclaw-combine/start_combine.sh  # 启动 Ray 训练集群
# 等 API 就绪后（脚本会提示）：
conda activate openclaw-rl
python experiment/train.py --exp-id exp27
```

---

## 详细步骤

### 第一步：停止当前实验

```bash
bash openclaw-combine/stop_combine.sh
```

脚本会：停止 Ray job → 停止 Ray 集群 → 杀残留进程 → 显示 GPU 显存。

确认所有 GPU `used` 降到 < 200 MiB 再继续。如果没降下来：

```bash
nvidia-smi
# 如果有残留进程：
kill -9 <PID>
```

### 第二步：清理旧实验数据

```bash
bash experiment/cleanup.sh <新实验ID>
# 例如：bash experiment/cleanup.sh exp26
```

这个脚本做三件事：
1. 归档当前的 `experiment_log.jsonl` 和 `eval_log.jsonl` 到 `exp{旧ID}_*.jsonl`
2. 清空 PRM record 文件
3. 删除旧 checkpoint（`/data/openclaw-rl/ckpt/qwen3-4b-openclaw-combine-full/iter_*`）

### 第三步：启动新实验

```bash
# 全量模式（batch=16，论文设置）
bash openclaw-combine/start_combine.sh

# 或 demo 模式（快速测试，5 分钟跑完）
bash openclaw-combine/start_combine.sh --demo
```

等到脚本输出 `API server ready` 或类似提示后，开始训练：

```bash
conda activate openclaw-rl
# 全量训练（约 1-2 小时）
python experiment/train.py --exp-id exp26

# 或 demo 模式
python experiment/train.py --demo --exp-id exp26
```

### 第四步：等待完成

训练会自动在 step 8 和 step 16 时触发评估。观察日志中的：
```
CHECKPOINT EVAL  tag=exp26_rl_step_8
```

训练完成后查看结果：
```bash
grep 'summary' experiment/results/eval_log.jsonl
```

---

## 常见问题

### OOM 崩溃
如果训练在 step 8-10 OOM，是因为 KL coef=0 导致策略漂移。当前 step 8 已能达到 0.95，可以接受。

### 评估分数异常低
检查是否清理了旧 checkpoint：
```bash
ls /data/openclaw-rl/ckpt/qwen3-4b-openclaw-combine-full/iter_*
```
如果有旧的 iter_* 目录（时间戳不是当前实验的），说明没有清理。

### 只想重新评估某个 checkpoint
```bash
conda activate openclaw-rl
python experiment/evaluate.py --tag exp26_manual --num-questions 36
```

### 想跑 baseline 评估（训练前）
在运行 train.py 之前：
```bash
python experiment/evaluate.py --tag exp26_baseline --num-questions 36
```
