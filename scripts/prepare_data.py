#!/usr/bin/env python3
"""
数据预处理脚本：将微信聊天记录转换为ShareGPT格式的训练数据
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def load_chat_data(input_path: str) -> List[Dict]:
    """加载原始聊天记录"""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_consecutive_messages(messages: List[Dict]) -> List[Dict]:
    """合并连续同一发送者的消息"""
    if not messages:
        return []

    merged = []
    current = {"sender": messages[0]["sender"], "content": messages[0]["content"]}

    for msg in messages[1:]:
        if msg["sender"] == current["sender"]:
            # 合并连续消息，用换行分隔
            current["content"] += "\n" + msg["content"]
        else:
            merged.append(current)
            current = {"sender": msg["sender"], "content": msg["content"]}

    merged.append(current)
    return merged


def convert_to_sharegpt(
    messages: List[Dict],
    my_name: str = "我",
    other_name: str = "对方",
    max_turns: int = 10
) -> List[Dict]:
    """
    将聊天记录转换为ShareGPT格式

    Args:
        messages: 合并后的消息列表
        my_name: "我"的标识符
        other_name: "对方"的标识符
        max_turns: 每个对话样本的最大轮次

    Returns:
        ShareGPT格式的对话列表
    """
    conversations_list = []
    current_conversation = []

    for msg in messages:
        sender = msg["sender"]
        content = msg["content"].strip()

        if not content:
            continue

        # 我 -> user, 对方 -> assistant
        if sender == my_name:
            role = "user"
        elif sender == other_name:
            role = "assistant"
        else:
            # 跳过未知发送者
            continue

        current_conversation.append({
            "role": role,
            "content": content
        })

        # 达到最大轮次时切分
        if len(current_conversation) >= max_turns * 2:
            # 确保以assistant结尾
            if current_conversation[-1]["role"] == "assistant":
                conversations_list.append({
                    "conversations": current_conversation.copy()
                })
            current_conversation = []

    # 处理剩余的对话
    if current_conversation:
        # 确保以assistant结尾，否则去掉最后一条user消息
        while current_conversation and current_conversation[-1]["role"] == "user":
            current_conversation.pop()

        # 确保以user开头
        while current_conversation and current_conversation[0]["role"] == "assistant":
            current_conversation.pop(0)

        if len(current_conversation) >= 2:
            conversations_list.append({
                "conversations": current_conversation
            })

    return conversations_list


def split_by_time_gap(
    messages: List[Dict],
    gap_threshold: int = 3600
) -> List[List[Dict]]:
    """
    按时间间隔切分对话（如果有时间戳字段）

    Args:
        messages: 原始消息列表
        gap_threshold: 时间间隔阈值（秒），超过则切分

    Returns:
        切分后的对话组列表
    """
    if not messages or "timestamp" not in messages[0]:
        return [messages]

    sessions = []
    current_session = [messages[0]]

    for i in range(1, len(messages)):
        prev_time = messages[i-1].get("timestamp", 0)
        curr_time = messages[i].get("timestamp", 0)

        if curr_time - prev_time > gap_threshold:
            sessions.append(current_session)
            current_session = []

        current_session.append(messages[i])

    if current_session:
        sessions.append(current_session)

    return sessions


def main():
    parser = argparse.ArgumentParser(description="将微信聊天记录转换为训练数据格式")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/chat.json",
        help="输入的聊天记录JSON文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/train_data.json",
        help="输出的训练数据JSON文件路径"
    )
    parser.add_argument(
        "--my-name",
        type=str,
        default="我",
        help="'我'在聊天记录中的标识符"
    )
    parser.add_argument(
        "--other-name",
        type=str,
        default="对方",
        help="'对方'在聊天记录中的标识符"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="每个训练样本的最大对话轮次"
    )
    parser.add_argument(
        "--time-gap",
        type=int,
        default=3600,
        help="对话切分的时间间隔阈值（秒）"
    )

    args = parser.parse_args()

    # 加载数据
    print(f"加载聊天记录: {args.input}")
    raw_messages = load_chat_data(args.input)
    print(f"原始消息数量: {len(raw_messages)}")

    # 按时间间隔切分对话
    sessions = split_by_time_gap(raw_messages, args.time_gap)
    print(f"对话会话数量: {len(sessions)}")

    # 转换每个会话
    all_conversations = []
    for session in sessions:
        # 合并连续消息
        merged = merge_consecutive_messages(session)

        # 转换为ShareGPT格式
        conversations = convert_to_sharegpt(
            merged,
            my_name=args.my_name,
            other_name=args.other_name,
            max_turns=args.max_turns
        )
        all_conversations.extend(conversations)

    print(f"生成训练样本数量: {len(all_conversations)}")

    # 统计信息
    total_turns = sum(len(c["conversations"]) for c in all_conversations)
    avg_turns = total_turns / len(all_conversations) if all_conversations else 0
    print(f"平均每个样本轮次: {avg_turns:.1f}")

    # 保存数据
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)

    print(f"训练数据已保存至: {args.output}")

    # 显示样本预览
    if all_conversations:
        print("\n=== 样本预览 ===")
        sample = all_conversations[0]["conversations"][:4]
        for turn in sample:
            role = "用户" if turn["role"] == "user" else "助手"
            content = turn["content"][:50] + "..." if len(turn["content"]) > 50 else turn["content"]
            print(f"[{role}] {content}")


if __name__ == "__main__":
    main()
