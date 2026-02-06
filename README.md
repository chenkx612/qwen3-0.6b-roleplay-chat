# 微信聊天角色扮演

使用 Qwen3-0.6B 模型，基于微信聊天记录进行微调，让模型学会模仿对话中对方的说话风格。

## 项目结构

```
roleplay/
├── data/
│   ├── chat.json              # 原始聊天记录（需要用户提供）
│   └── train_data.json        # 预处理后的训练数据
├── scripts/
│   ├── prepare_data.py        # 数据预处理脚本
│   └── convert_to_gguf.py     # 模型转换脚本（可选）
├── train/
│   └── finetune.ipynb         # Colab微调notebook
├── inference/
│   └── chat.py                # 本地推理脚本
├── requirements.txt           # 依赖
└── README.md                  # 说明文档
```

## 使用流程

### 1. 准备聊天数据

将微信聊天记录整理为JSON格式，保存到 `data/chat.json`：

```json
[
  {"sender": "我", "content": "在干嘛"},
  {"sender": "对方", "content": "刚下班回来"},
  {"sender": "我", "content": "今天累不累"},
  {"sender": "对方", "content": "还好吧，开了一天会"}
]
```

字段说明：
- `sender`: 发送者标识，"我"或"对方"
- `content`: 消息内容
- `timestamp`: （可选）时间戳，用于切分对话会话

### 2. 数据预处理

```bash
python scripts/prepare_data.py -i data/chat.json -o data/train_data.json
```

参数说明：
- `--input, -i`: 输入的聊天记录文件
- `--output, -o`: 输出的训练数据文件
- `--my-name`: "我"的标识符（默认："我"）
- `--other-name`: "对方"的标识符（默认："对方"）
- `--max-turns`: 每个样本的最大对话轮次（默认：10）
- `--time-gap`: 对话切分的时间间隔阈值，秒（默认：3600）

### 3. 模型微调（Google Colab）

1. 打开 [Google Colab](https://colab.research.google.com/)
2. 上传 `train/finetune.ipynb`
3. 选择 GPU 运行时（运行时 → 更改运行时类型 → T4 GPU）
4. 按顺序运行所有单元格
5. 上传 `data/train_data.json` 作为训练数据
6. 训练完成后下载 `lora_adapter.zip`

### 4. 本地推理

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

### 自定义聊天记录格式

如果你的聊天记录使用不同的字段名，修改 `prepare_data.py` 中的字段映射：

```python
# 示例：字段名为 "from" 和 "text"
msg["sender"] = raw_msg["from"]
msg["content"] = raw_msg["text"]
```

## 注意事项

1. **数据隐私**: 聊天记录可能包含敏感信息，请妥善保管
2. **训练数据量**: 建议至少 100 条以上的对话记录，效果更好
3. **显存需求**: Colab 免费版 T4 GPU（16GB）足够训练
4. **推理速度**: CPU 推理较慢，建议使用 llama.cpp 或量化模型

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
