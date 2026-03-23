# OpenClaw-RL 部署与训练流程文档

> 环境：GPU-zxy2-test6 (8×A100-40GB, Ubuntu 22.04, CUDA Driver 12.8)
> 训练场景：Tool-call RL (数学推理 + Python 工具调用)
> 训练方式：LoRA + FSDP (无需 transformer_engine)
> 模型：Qwen3-4B
> 数据：DAPO-Math-17k

---

## 一、服务器信息

```
Host:     js2.blockelite.cn
Port:     25900
User:     test6
GPU:      8 × NVIDIA A100-SXM4-40GB
OS:       Ubuntu 22.04.5 LTS
Driver:   NVIDIA 570.158.01 (CUDA 12.8)
```

---

## 二、环境配置步骤

### 2.1 创建 Conda 环境

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n openclaw-rl python=3.12 -y
conda activate openclaw-rl
```

### 2.2 配置国内镜像源

服务器到 PyPI/GitHub 的速度极慢（~200KB/s），必须使用国内镜像：

```bash
# pip 默认使用清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

### 2.3 安装 PyTorch

```bash
# 清华镜像有完整的 PyTorch wheel
# 注意：服务器 CUDA Driver 570.x 支持 CUDA 12.8，cu126 兼容
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证：
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# 输出: 2.9.1+cu126 True 8
```

### 2.4 安装 Python 依赖

requirements.txt 中有 ~270 个包，部分需要特殊处理：

```bash
cd ~/OpenClaw-RL

# 过滤掉系统级包和 git 源包
grep -v "^-e git\|^git+\|torch==\|torchvision==\|torchaudio==\|nvidia-\|triton==\|^devscripts\|mbridge\|megatron-bridge\|megatron_core\|torch_memory_saver\|transformer_engine\|dbus-python\|PyGObject\|launchpadlib\|lazr\.\|cubloaty\|apache-tvm\|nixl" requirements.txt > /tmp/requirements_filtered.txt

pip install -r /tmp/requirements_filtered.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.5 安装核心框架（本地源码）

```bash
# slime (RL 训练框架)
pip install -e ~/OpenClaw-RL/slime

# Megatron-LM (分布式训练引擎)
pip install -e ~/OpenClaw-RL/Megatron-LM
```

### 2.6 安装 Git 依赖（通过 GitHub 代理）

由于 GitHub 直连极慢，使用 `gh-proxy.com` 代理下载 zip 后本地安装：

```bash
cd /tmp

# megatron-bridge
curl -L -o mb.zip "https://gh-proxy.com/https://github.com/fzyzcjy/Megatron-Bridge/archive/35b4ebfc486fb15dcc0273ceea804c3606be948a.zip"
unzip -qo mb.zip
pip install -e ./Megatron-Bridge-35b4ebfc486fb15dcc0273ceea804c3606be948a \
    --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple

# torch_memory_saver (PyPI 有预编译版本)
pip install torch_memory_saver -i https://pypi.tuna.tsinghua.edu.cn/simple

# sglang (推理引擎)
curl -L -o sglang.zip "https://gh-proxy.com/https://github.com/sgl-project/sglang/archive/dce8b0606c06d3a191a24c7b8cbe8e238ab316c9.zip"
unzip -qo sglang.zip
pip install ./sglang-dce8b0606c06d3a191a24c7b8cbe8e238ab316c9/python \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# mbridge
pip install "mbridge @ git+https://gh-proxy.com/https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.7 安装 CUDA Toolkit 和编译型依赖

```bash
# CUDA Toolkit (flashinfer JIT 编译需要 nvcc)
conda install -c nvidia cuda-toolkit=12.6 -y

# flash-attn (从源码编译，需要 nvcc)
pip install flash-attn --no-build-isolation

# apex (仅 Python 部分，不编译 CUDA 扩展)
cd /tmp
curl -L -o apex.zip "https://gh-proxy.com/https://github.com/NVIDIA/apex/archive/refs/heads/master.zip"
unzip -qo apex.zip
cd apex-master
pip install . --no-build-isolation

# transformer_engine (仅 Python + CUDA 预编译部分)
pip install transformer_engine==2.10.0 transformer_engine_cu12==2.10.0 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**注意：`transformer_engine_torch` 需要完整 CUDA 开发工具链才能编译，在此环境中跳过。这不影响 LoRA + FSDP 训练。**

### 2.8 修复 flashinfer 链接器问题

conda 安装的 CUDA toolkit 把库文件放在 `lib/`，但 flashinfer 的 ninja 编译脚本搜索 `lib64/`：

```bash
CONDA_PREFIX=$HOME/anaconda3/envs/openclaw-rl
mkdir -p $CONDA_PREFIX/lib64/stubs
ln -sf $CONDA_PREFIX/lib/libcudart.so $CONDA_PREFIX/lib64/libcudart.so
ln -sf $CONDA_PREFIX/lib/stubs/libcuda.so $CONDA_PREFIX/lib64/stubs/libcuda.so
```

### 2.9 预热 flashinfer JIT 缓存

首次运行 SGLang 时，flashinfer 需要 JIT 编译 CUDA kernel。可提前预热：

```bash
python -c "
import flashinfer, torch
q = torch.randn(32, 128, dtype=torch.bfloat16, device='cuda')
k = torch.randn(32, 128, dtype=torch.bfloat16, device='cuda')
v = torch.randn(32, 128, dtype=torch.bfloat16, device='cuda')
flashinfer.single_prefill_with_kv_cache(q, k, v)
print('flashinfer JIT warmup OK')
"
```

### 2.10 验证核心包

```bash
python -c "
import torch; print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
import transformers; print(f'transformers: {transformers.__version__}')
import sglang; print('sglang: OK')
import ray; print(f'ray: {ray.__version__}')
import flash_attn; print(f'flash_attn: {flash_attn.__version__}')
import megatron; print('megatron: OK')
import slime; print('slime: OK')
import peft; print('peft: OK')
from torch.distributed.fsdp import FullyShardedDataParallel; print('FSDP: OK')
print('ALL CORE PACKAGES OK')
"
```

---

## 三、数据准备

### 3.1 下载模型

```bash
# 使用 HF 国内镜像
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-4B --local-dir ~/models/Qwen3-4B
```

模型大小约 7.6GB。

### 3.2 下载并预处理训练数据

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "
from datasets import load_dataset
import os, json

os.makedirs(os.path.expanduser('~/data'), exist_ok=True)
ds = load_dataset('BytedTsinghua-SIA/DAPO-Math-17k', split='train')

out = os.path.expanduser('~/data/dapo_math_17k.jsonl')
with open(out, 'w') as f:
    count = 0
    for ex in ds:
        if count >= 17000:  # 只取前 17k 条
            break
        prompt = ex['prompt'][0]['content'] if ex['prompt'] else ''
        label = ex['reward_model']['ground_truth']
        f.write(json.dumps({'prompt': prompt, 'label': label}) + '\n')
        count += 1
print(f'Saved {count} samples to {out}')
"
```

---

## 四、训练脚本说明

训练脚本位于 `~/OpenClaw-RL/toolcall-rl/retool_qwen3_4b_rl_lora.sh`。

### 4.1 GPU 分配

```
总 GPU: 8 × A100-40GB
├── Actor (LoRA 训练): 4 GPU (FSDP)
└── Rollout (SGLang 推理): 4 GPU (2 个引擎 × TP=2)
```

### 4.2 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--train-backend` | fsdp | 使用 FSDP 后端（不依赖 Megatron TE） |
| `--use-lora` | - | 启用 LoRA 微调 |
| `--lora-rank` | 16 | LoRA 秩 |
| `--lora-alpha` | 32 | LoRA 缩放因子 |
| `--rollout-batch-size` | 8 | 每步收集 8 个 prompt 的样本 |
| `--n-samples-per-prompt` | 4 | 每个 prompt 采样 4 次（GRPO 需要） |
| `--rollout-temperature` | 1.0 | 采样温度 |
| `--lr` | 1e-5 | 学习率 |
| `--kl-loss-coef` | 0.01 | KL 散度惩罚系数 |
| `--eps-clip` | 0.2 | PPO 下界裁剪 |
| `--eps-clip-high` | 0.28 | PPO 上界裁剪（非对称） |
| `--advantage-estimator` | grpo | 使用 GRPO 优势估计 |
| `--rollout-max-response-len` | 8192 | 最大回复长度 |

### 4.3 Toolcall-RL 的工作流

1. **Rollout 阶段**：SGLang 引擎加载 Qwen3-4B，对每个数学题生成回复
2. **工具调用**：模型生成 `<tool_call>` 标记时，Python 沙箱执行代码
3. **奖励计算**：检查最终答案是否正确（`Answer: \boxed{...}` 与 ground truth 比较）
4. **GRPO 训练**：基于奖励信号，使用 PPO-style 损失更新 LoRA 权重
5. **权重同步**：训练完成后广播新权重到 SGLang 引擎

### 4.4 安全沙箱

Python 代码执行使用安全沙箱（`tool_sandbox.py`）：
- 白名单模块：math, random, datetime, collections, itertools 等
- 禁止的操作：os, sys, subprocess, eval, exec, open, __import__
- 超时限制：120 秒
- 内存限制：4GB
- 最大并发：32 进程

---

## 五、启动训练

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openclaw-rl
cd ~/OpenClaw-RL/slime

# 启动训练（后台运行）
PROMPT_DATA=$HOME/data/dapo_math_17k.jsonl \
nohup bash ../toolcall-rl/retool_qwen3_4b_rl_lora.sh > ~/train_toolcall_rl.log 2>&1 &
```

### 5.1 监控训练

```bash
# 查看实时日志
tail -f ~/train_toolcall_rl.log

# 查看训练关键指标
grep -E "actor_train|train end|rollout_time|Final collected|acc" ~/train_toolcall_rl.log

# 查看 GPU 使用
nvidia-smi
```

### 5.2 训练速度参考

在 8×A100-40GB 上的实际观测：

| 阶段 | 耗时 |
|------|------|
| Rollout (32 样本) | ~120 秒 |
| 训练 (1 step) | ~15-35 秒 |
| 权重更新 | ~3 秒 |
| **总 step 时间** | **~190 秒** |

### 5.3 检查点保存

每 20 个训练步保存一次检查点到 `~/ckpt/qwen3-4b-retool-rl-lora/`。

---

## 六、遇到的问题与解决方案

### 问题 1：PyTorch 官方源极慢

**现象**：从 `download.pytorch.org` 下载速度仅 ~200B/s
**解决**：使用清华镜像 `https://pypi.tuna.tsinghua.edu.cn/simple`

### 问题 2：GitHub 依赖下载超慢

**现象**：`pip install git+https://github.com/...` 卡住
**解决**：通过 `gh-proxy.com` 代理下载 zip 后本地安装

### 问题 3：transformer_engine_torch 编译失败

**现象**：需要完整 CUDA 开发工具链
**解决**：跳过此包，改用 FSDP 后端（`--train-backend fsdp`）。
LoRA 训练不依赖 Megatron 的 Transformer Engine。

### 问题 4：flashinfer JIT 编译链接失败

**现象**：`ld: cannot find -lcudart: No such file or directory`
**原因**：conda 安装的 CUDA toolkit 把 `.so` 放在 `lib/`，但编译脚本搜索 `lib64/`
**解决**：创建符号链接
```bash
ln -sf $CONDA_PREFIX/lib/libcudart.so $CONDA_PREFIX/lib64/libcudart.so
ln -sf $CONDA_PREFIX/lib/stubs/libcuda.so $CONDA_PREFIX/lib64/stubs/libcuda.so
```

### 问题 5：Ray worker 进程找不到 nvcc

**现象**：SGLang 引擎内 flashinfer JIT 编译找不到 nvcc
**解决**：在训练脚本的 `RUNTIME_ENV_JSON` 中注入 `PATH` 和 `CUDA_HOME`

### 问题 6：磁盘空间紧张

**现象**：根分区仅 437GB，安装依赖后仅剩 ~28GB
**解决**：
- `pip cache purge && conda clean --all -y`
- 删除临时编译文件
- 训练数据只取 17k 条（而非完整 1.79M 条）

---

## 七、目录结构

```
~/OpenClaw-RL/
├── toolcall-rl/                    # 工具调用 RL 代码
│   ├── generate_with_retool.py     # 自定义 rollout + reward 函数
│   ├── tool_sandbox.py             # Python 安全沙箱
│   └── retool_qwen3_4b_rl_lora.sh # LoRA 训练脚本 (本次使用)
├── slime/                          # RL 训练框架
│   └── train_async.py              # 异步训练入口
├── Megatron-LM/                    # 分布式训练引擎
└── ...

~/models/Qwen3-4B/                  # 模型权重 (~7.6GB)
~/data/dapo_math_17k.jsonl          # 训练数据 (17k 条)
~/ckpt/qwen3-4b-retool-rl-lora/     # 训练检查点
~/train_toolcall_rl.log             # 训练日志
```
