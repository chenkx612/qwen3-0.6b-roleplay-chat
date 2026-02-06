# 微信聊天角色扮演

使用 Qwen3-0.6B 模型，基于微信聊天记录进行微调，让模型学会模仿对话中对方的说话风格。

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

1. 打开 [Google Colab](https://colab.research.google.com/)
2. 上传 `train/finetune.ipynb`
3. 选择 GPU 运行时（运行时 → 更改运行时类型 → T4 GPU）
4. 按顺序运行所有单元格
5. 上传 `data/train_data.json` 作为训练数据
6. 训练完成后下载 `lora_adapter.zip`

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
