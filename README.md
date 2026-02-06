# 微信聊天角色扮演

使用 Qwen3-0.6B 模型 + LoRA 微调，基于微信聊天记录让模型学会模仿对话中对方的说话风格。

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 基座模型 | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 阿里通义千问，6亿参数，中文能力强 |
| 微调方法 | LoRA | 低秩适配，仅训练1.3%参数 |
| 量化方案 | QLoRA (4-bit NF4) | 显存占用低，免费Colab可跑 |
| 训练框架 | TRL + PEFT | HuggingFace官方SFT训练库 |
| 推理后端 | Transformers / llama.cpp | 支持CPU推理 |

## 项目结构

```
roleplay/
├── data/
│   └── train_data.json        # 训练数据（ShareGPT格式，手写）
├── scripts/
│   └── convert_to_gguf.py     # 模型转换脚本（可选）
├── train/
│   └── finetune.ipynb         # Colab微调notebook
├── inference/
│   └── chat.py                # 本地推理脚本
├── requirements.txt           # 依赖
└── README.md                  # 说明文档
```

## 使用流程

### 1. 准备训练数据

直接手写 `data/train_data.json`，使用 ShareGPT 格式：

```json
[
  {
    "conversations": [
      {"role": "user", "content": "在干嘛"},
      {"role": "assistant", "content": "刚下班回来"},
      {"role": "user", "content": "今天累不累"},
      {"role": "assistant", "content": "还好吧，开了一天会"}
    ]
  },
  {
    "conversations": [
      {"role": "user", "content": "晚上吃什么"},
      {"role": "assistant", "content": "还没想好，可能点个外卖"}
    ]
  }
]
```

格式要求：
- `user`: 你发的消息
- `assistant`: 对方的回复（模型要学习的目标）
- 每个对话必须以 `user` 开头、`assistant` 结尾
- 可以包含多轮对话，按时间顺序排列
- 建议将相关的连续对话组织在同一个 `conversations` 数组中

### 2. 模型微调（Google Colab）

#### 快速开始

1. 打开 [Google Colab](https://colab.research.google.com/)
2. 上传 `train/finetune.ipynb`
3. 选择 GPU 运行时（运行时 → 更改运行时类型 → T4 GPU）
4. 按顺序运行所有单元格
5. 上传 `data/train_data.json` 作为训练数据
6. 训练完成后下载 `lora_adapter.zip`

#### 微调技术细节

**LoRA 配置**

```python
LoraConfig(
    r=16,                    # LoRA秩，控制低秩矩阵的维度
    lora_alpha=32,           # 缩放系数，alpha/r = 2 为常用设置
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层的QKV和输出投影
        "gate_proj", "up_proj", "down_proj"       # FFN层的门控和上下投影
    ],
    lora_dropout=0.05,       # Dropout防止过拟合
    bias="none",             # 不训练bias
    task_type="CAUSAL_LM"    # 因果语言模型任务
)
```

- **可训练参数**: 10,092,544 (约10M)
- **总参数量**: 761,724,928 (约762M)
- **可训练比例**: 1.325%

**4-bit 量化配置 (QLoRA)**

```python
BitsAndBytesConfig(
    load_in_4bit=True,           # 4-bit量化加载
    bnb_4bit_quant_type="nf4",   # NormalFloat4量化类型
    bnb_4bit_compute_dtype=torch.float16,  # 计算时使用FP16
    bnb_4bit_use_double_quant=True  # 双重量化进一步压缩
)
```

**训练超参数**

| 参数 | 值 | 说明 |
|------|-----|------|
| num_train_epochs | 2 | 训练轮数 |
| per_device_train_batch_size | 4 | 单卡批次大小 |
| gradient_accumulation_steps | 4 | 梯度累积步数 |
| effective_batch_size | 16 | 等效批次大小 (4×4) |
| learning_rate | 2e-4 | 学习率 |
| lr_scheduler_type | cosine | 余弦学习率调度 |
| warmup_ratio | 0.1 | 预热比例 |
| optimizer | paged_adamw_8bit | 8-bit分页优化器 |
| max_seq_length | 512 | 最大序列长度 |

**数据格式化**

训练数据使用 Qwen3 的 ChatML 模板格式化：

```
<|im_start|>user
在干嘛<|im_end|>
<|im_start|>assistant
刚下班回来<|im_end|>
```

**输出文件**

- `lora_adapter/` - LoRA适配器权重（约40MB）
  - `adapter_model.safetensors` - LoRA权重
  - `adapter_config.json` - LoRA配置
  - `tokenizer.json` - 分词器
- `merged_model/` - 合并后的完整模型（约1.5GB）

### 3. 本地推理

安装依赖：

```bash
pip install -r requirements.txt
```

运行聊天：

```bash
# 使用LoRA适配器（推荐）
python inference/chat.py --model Qwen/Qwen3-0.6B --lora ./lora_adapter

# 使用合并后的完整模型
python inference/chat.py --model ./merged_model

# 添加系统提示
python inference/chat.py --model Qwen/Qwen3-0.6B --lora ./lora_adapter \
    --system-prompt "你是一个温柔体贴的朋友"
```

交互示例：

```
>>> 在干嘛
刚下班回来，今天有点累
>>> 晚上吃什么
还没想好，可能点个外卖吧
>>> /quit
再见！
```

命令：
- `/quit` 或 `/exit`: 退出程序
- `/clear`: 清空对话历史

## 高级用法

### 使用 llama.cpp 加速推理

1. 安装 llama-cpp-python：

```bash
pip install llama-cpp-python
```

2. 转换模型为GGUF格式（参考 `scripts/convert_to_gguf.py` 中的说明）

3. 运行：

```bash
python inference/chat.py --backend llama.cpp --gguf ./model.gguf
```

## 注意事项

1. **数据隐私**: 聊天记录可能包含敏感信息，请妥善保管
2. **训练数据量**: 建议至少 100 条以上的对话记录，效果更好
3. **显存需求**: Colab 免费版 T4 GPU（16GB）足够训练
4. **推理速度**: CPU 推理较慢，建议使用 llama.cpp 或量化模型

## 调参指南

### 数据量 vs 训练轮数

| 数据量 | 建议epochs | 说明 |
|--------|------------|------|
| < 50条 | 3-5 | 数据少需要更多轮次 |
| 50-200条 | 2-3 | 默认设置 |
| > 200条 | 1-2 | 数据充足，防止过拟合 |

### LoRA 参数调整

| 场景 | r值 | lora_alpha | 说明 |
|------|-----|------------|------|
| 数据少，轻度适配 | 8 | 16 | 参数少，泛化好 |
| 默认设置 | 16 | 32 | 平衡效果 |
| 数据多，深度学习 | 32 | 64 | 参数多，拟合强 |

### 学习率调整

- **默认**: 2e-4
- **过拟合（loss震荡）**: 降至 1e-4 或 5e-5
- **欠拟合（loss下降慢）**: 升至 3e-4 或 5e-4

### 常见问题诊断

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| 回复过于模板化 | 数据重复、epochs过多 | 增加数据多样性，减少epochs |
| 回复偏离风格 | 数据量不足、epochs过少 | 增加数据或epochs |
| 回复无意义 | 学习率过高、数据质量差 | 降低学习率，清洗数据 |
| 显存OOM | batch_size过大 | 减小batch_size，增加gradient_accumulation |

## 常见问题

**Q: 模型回复不像对方的风格？**

A: 可能原因：
- 训练数据量不足
- 训练轮数不够（尝试增加 epochs）
- 对话样本质量问题（过短或无意义的对话）

**Q: 本地推理很慢？**

A: 尝试：
- 使用 llama.cpp 后端
- 使用量化模型（q4_k_m）
- 减少 max_new_tokens

**Q: Colab 训练中断？**

A:
- 保存检查点后可以继续训练
- 使用 Colab Pro 获得更长运行时间

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Qwen3 技术报告](https://qwenlm.github.io/blog/qwen3/)
- [HuggingFace PEFT 文档](https://huggingface.co/docs/peft)
- [TRL SFTTrainer 文档](https://huggingface.co/docs/trl/sft_trainer)
