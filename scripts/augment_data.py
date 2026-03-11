#!/usr/bin/env python3
"""
数据增强脚本：分析现有对话风格，调用 LLM API 批量生成新对话
兼容所有 OpenAI 格式 API（DeepSeek、OpenAI、本地模型等）
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("错误：需要安装 openai 包")
    print("请运行：pip install openai")
    sys.exit(1)


def load_dotenv(env_path: str = None):
    """从 .env 文件加载环境变量（不覆盖已有值）"""
    candidates = [env_path] if env_path else [
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    for p in candidates:
        p = Path(p)
        if p.is_file():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = value
            return


# ============================================================
# 话题池
# ============================================================

TOPIC_POOL = [
    "约吃饭/聚餐",
    "天气变化",
    "工作或学习上的吐槽",
    "网购/快递到了",
    "打游戏/开黑",
    "追剧/综艺推荐",
    "旅行/出游计划",
    "运动/健身",
    "节日祝福/节日安排",
    "日常吐槽/抱怨",
    "借东西/帮忙",
    "早安/晚安闲聊",
    "生病/身体不舒服",
    "周末计划",
    "搬家/租房",
    "拍照/修图/发朋友圈",
    "宠物",
    "做饭/点外卖",
    "考试/作业/论文",
    "约看电影",
    "发红包/转账",
    "推荐歌曲/音乐",
    "讨论某个朋友/共同认识的人",
    "吐槽某件尴尬的事",
    "问路/问地址",
    "约逛街/购物",
    "讨论发型/穿搭",
    "讨论某个新闻/热搜",
    "请教问题/求推荐",
    "分享搞笑的事/段子",
    "讨论放假安排",
    "约打球/跑步",
    "回忆以前的事",
    "送礼物/选礼物",
    "讨论某家店/某个地方",
    "通知/传达消息",
    "还钱/AA制",
    "迟到/爽约道歉",
    "安慰/鼓励对方",
    "讨论毕业/升学/找工作",
    "约KTV/唱歌",
    "分享照片/视频",
    "讨论搞笑视频/表情包",
    "约自习/一起学习",
    "讨论某个APP/软件",
    "问对方在干嘛",
    "讨论家人/家里的事",
    "聊睡眠/熬夜",
    "讨论某个老师/同事",
]


# ============================================================
# 阶段1：加载与校验
# ============================================================

def load_and_validate(path: str) -> list:
    """加载并校验 ShareGPT 格式的训练数据"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("错误：输入文件应为 JSON 数组")
        sys.exit(1)

    total_turns = 0
    for i, item in enumerate(data):
        if "conversations" not in item:
            print(f"错误：第 {i+1} 组对话缺少 conversations 字段")
            sys.exit(1)
        convs = item["conversations"]
        if not isinstance(convs, list) or len(convs) < 2:
            print(f"错误：第 {i+1} 组对话轮数不足")
            sys.exit(1)
        for j, msg in enumerate(convs):
            if "role" not in msg or "content" not in msg:
                print(f"错误：第 {i+1} 组对话第 {j+1} 条消息缺少 role/content")
                sys.exit(1)
        total_turns += len(convs)

    avg_turns = total_turns / len(data) if data else 0
    print(f"输入数据：{len(data)} 组对话，共 {total_turns} 条消息，平均 {avg_turns:.1f} 条/组")
    return data


# ============================================================
# 阶段2：风格分析
# ============================================================

def analyze_style(data: list) -> dict:
    """统计分析 assistant 消息的风格特征"""
    assistant_msgs = []
    for item in data:
        for msg in item["conversations"]:
            if msg["role"] == "assistant":
                assistant_msgs.append(msg["content"])

    if not assistant_msgs:
        print("警告：未找到 assistant 消息")
        return {}

    # 消息长度
    lengths = [len(m) for m in assistant_msgs]
    avg_len = sum(lengths) / len(lengths)
    sorted_lengths = sorted(lengths)
    median_len = sorted_lengths[len(sorted_lengths) // 2]

    # 连发消息（\n 分隔）
    multi_msgs = [m for m in assistant_msgs if "\n" in m]
    multi_ratio = len(multi_msgs) / len(assistant_msgs)
    sub_counts = [len(m.split("\n")) for m in multi_msgs] if multi_msgs else [1]
    avg_sub = sum(sub_counts) / len(sub_counts)

    # 所有子消息（按 \n 拆分）
    all_sub_msgs = []
    for m in assistant_msgs:
        all_sub_msgs.extend(m.split("\n"))
    all_sub_msgs = [s.strip() for s in all_sub_msgs if s.strip()]

    # 句尾特征
    ending_chars = []
    for s in all_sub_msgs:
        if s:
            ending_chars.append(s[-1])
    ending_counter = Counter(ending_chars)
    # 高频句尾（非标点的语气词尾）
    tail_particles = ["不", "嘛", "呀", "吧", "啦", "呢", "哈", "的", "滴", "了", "噢", "哦", "嗯"]
    tail_stats = {}
    for p in tail_particles:
        count = sum(1 for s in all_sub_msgs if s.endswith(p))
        if count > 0:
            tail_stats[p] = round(count / len(all_sub_msgs), 2)

    # 标点习惯
    total_chars = sum(len(m) for m in assistant_msgs)
    punct_stats = {}
    for p, name in [("。", "句号"), ("！", "感叹号"), ("？", "问号"), ("，", "逗号"),
                     ("…", "省略号"), ("~", "波浪号")]:
        count = sum(m.count(p) for m in assistant_msgs)
        if count > 0:
            punct_stats[name] = count
    uses_period = sum(m.count("。") for m in assistant_msgs)

    # 口头禅/高频短语（2-6字）
    phrase_counter = Counter()
    for m in all_sub_msgs:
        # 提取2-6字的子串
        for length in range(2, 7):
            for start in range(len(m) - length + 1):
                substr = m[start:start + length]
                # 过滤纯标点
                if any('\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf' for c in substr):
                    phrase_counter[substr] += 1

    # 跨对话出现的短语才算口头禅
    catchphrases = []
    for phrase, count in phrase_counter.most_common(200):
        if count < 2:
            break
        # 检查是否跨对话出现
        conv_count = 0
        for item in data:
            for msg in item["conversations"]:
                if msg["role"] == "assistant" and phrase in msg["content"]:
                    conv_count += 1
                    break
        if conv_count >= 2:
            # 排除子串关系：如果已有更长的包含此短语，跳过
            is_sub = False
            for existing in catchphrases:
                if phrase in existing and phrase != existing:
                    is_sub = True
                    break
            if not is_sub:
                # 移除已有的更短子串
                catchphrases = [e for e in catchphrases if e not in phrase or e == phrase]
                catchphrases.append(phrase)
        if len(catchphrases) >= 15:
            break

    # 字符替换习惯
    char_replacements = {}
    replacement_patterns = [
        ("蟹蟹", "谢谢"), ("滴", "的"), ("俺", "我"), ("嘛", "吗"),
        ("不", "不（语气词）"), ("哈哈哈", "笑声"),
    ]
    for variant, standard in replacement_patterns:
        count = sum(m.count(variant) for m in assistant_msgs)
        if count > 0:
            char_replacements[variant] = {"含义": standard, "出现次数": count}

    # 语气标记
    tone_markers = {}
    patterns = [
        ("哈哈", "笑声"), ("不好意思", "道歉"), ("抱歉", "道歉"),
        ("可能", "犹豫"), ("应该", "犹豫"), ("感觉", "犹豫"),
        ("要不", "建议/犹豫"),
    ]
    for marker, category in patterns:
        count = sum(m.count(marker) for m in assistant_msgs)
        if count > 0:
            tone_markers[marker] = {"类型": category, "次数": count}

    # 是否反问
    question_msgs = sum(1 for s in all_sub_msgs if "?" in s or "？" in s or
                        s.endswith("不") or s.endswith("吗") or s.endswith("嘛") or
                        s.endswith("呢"))
    question_ratio = round(question_msgs / len(all_sub_msgs), 2) if all_sub_msgs else 0

    analysis = {
        "消息统计": {
            "总assistant消息数": len(assistant_msgs),
            "平均长度": round(avg_len, 1),
            "中位数长度": median_len,
        },
        "连发消息": {
            "连发比例": round(multi_ratio, 2),
            "平均子消息数": round(avg_sub, 1),
        },
        "口头禅": catchphrases,
        "字符替换": char_replacements,
        "句尾语气词频率": tail_stats,
        "标点习惯": {
            "使用句号": uses_period > 0,
            "各标点统计": punct_stats,
        },
        "语气标记": tone_markers,
        "反问比例": question_ratio,
    }

    print("\n===== 风格分析结果 =====")
    print(f"  口头禅: {', '.join(catchphrases[:10]) if catchphrases else '无明显口头禅'}")
    print(f"  字符替换: {', '.join(f'{k}→{v['含义']}' for k, v in char_replacements.items()) if char_replacements else '无'}")
    print(f"  平均消息长度: {avg_len:.1f} 字符，中位数: {median_len}")
    print(f"  连发消息比例: {multi_ratio:.0%}，平均子消息数: {avg_sub:.1f}")
    print(f"  反问比例: {question_ratio:.0%}")
    if tail_stats:
        top_tails = sorted(tail_stats.items(), key=lambda x: -x[1])[:5]
        print(f"  高频句尾: {', '.join(f'{k}({v:.0%})' for k, v in top_tails)}")
    print()

    return analysis


# ============================================================
# 阶段3：构建 Prompt
# ============================================================

def build_system_prompt(analysis: dict, sample_conversations: list, persona: str = None) -> str:
    """将风格分析转化为生成指令"""
    parts = ["你是一个对话数据生成器。你需要模仿以下说话风格，生成微信日常聊天对话。"]
    parts.append("")

    if persona:
        parts.append("## 角色人设")
        parts.append("")
        parts.append(f"你要模仿的对方的人设：{persona}")
        parts.append("")

    # 风格描述
    parts.append("## 说话风格要求")
    parts.append("")

    catchphrases = analysis.get("口头禅", [])
    if catchphrases:
        parts.append(f"- 口头禅和高频表达：{'、'.join(catchphrases[:10])}")

    char_rep = analysis.get("字符替换", {})
    if char_rep:
        reps = [f"用「{k}」代替「{v['含义']}」" for k, v in char_rep.items()
                if v['含义'] not in ("笑声", "不（语气词）")]
        if reps:
            parts.append(f"- 用字习惯：{'，'.join(reps)}")

    msg_stats = analysis.get("消息统计", {})
    avg_len = msg_stats.get("平均长度", 20)
    parts.append(f"- 消息长度：平均约 {avg_len:.0f} 个字符，保持简短口语化")

    multi = analysis.get("连发消息", {})
    multi_ratio = multi.get("连发比例", 0)
    avg_sub = multi.get("平均子消息数", 1)
    if multi_ratio > 0.3:
        parts.append(f"- 经常连发多条消息（约 {multi_ratio:.0%} 的回复包含多条），用换行符 \\n 分隔子消息，平均 {avg_sub:.1f} 条")
    elif multi_ratio > 0:
        parts.append(f"- 偶尔连发消息（用 \\n 分隔），比例约 {multi_ratio:.0%}")

    tail_stats = analysis.get("句尾语气词频率", {})
    if tail_stats:
        top = sorted(tail_stats.items(), key=lambda x: -x[1])[:5]
        parts.append(f"- 常用句尾语气：{'、'.join(f'{k}' for k, _ in top)}")

    punct = analysis.get("标点习惯", {})
    if not punct.get("使用句号", True):
        parts.append("- 几乎不使用句号")

    tone = analysis.get("语气标记", {})
    if tone:
        markers = list(tone.keys())
        parts.append(f"- 常用语气词/表达：{'、'.join(markers[:8])}")

    q_ratio = analysis.get("反问比例", 0)
    if q_ratio > 0.2:
        parts.append(f"- 经常在回复中反问对方（约 {q_ratio:.0%} 的子消息含问句）")

    # 示例对话
    parts.append("")
    parts.append("## 示例对话（参考风格，不要复制内容）")
    parts.append("")
    for i, conv in enumerate(sample_conversations[:3]):
        parts.append(f"示例 {i+1}:")
        for msg in conv["conversations"]:
            role_label = "用户" if msg["role"] == "user" else "对方"
            parts.append(f"  {role_label}: {msg['content']}")
        parts.append("")

    # 输出格式
    parts.append("## 输出格式要求")
    parts.append("")
    parts.append("严格输出 JSON 数组，格式如下：")
    parts.append("""```
[
  {
    "conversations": [
      {"role": "user", "content": "用户说的话"},
      {"role": "assistant", "content": "对方的回复"},
      ...
    ]
  },
  ...
]
```""")
    parts.append("")
    parts.append("规则：")
    parts.append("- 每组对话必须以 user 开头、assistant 结尾")
    parts.append("- user 和 assistant 严格交替")
    parts.append("- 每组对话 4-10 条消息")
    parts.append("- assistant 的回复要自然、口语化，符合上述风格")
    parts.append("- user 的消息简短随意即可")
    parts.append("- 连发消息用 \\n 分隔（如：\"到啦\\n你们呢\"）")
    parts.append("- 只输出 JSON，不要输出其他内容")

    return "\n".join(parts)


def build_user_prompt(topics: list, batch_size: int) -> str:
    """构建用户 prompt，指定生成话题"""
    topic_list = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(topics))
    return f"""请生成 {batch_size} 组微信聊天对话，话题方向分别是：
{topic_list}

每组对话 4-10 条消息，严格按照 JSON 格式输出。"""


# ============================================================
# 阶段4：API 调用
# ============================================================

def parse_json_response(text: str) -> list:
    """从 LLM 响应中提取 JSON 数组"""
    # 尝试直接解析
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        # json_object 模式可能返回 {"conversations": [...]} 或类似结构
        if isinstance(result, dict):
            for key in ("data", "conversations", "dialogues", "result"):
                if key in result and isinstance(result[key], list):
                    return result[key]
    except json.JSONDecodeError:
        pass

    # 去掉 markdown 代码块
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    if cleaned != text:
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # regex 提取最外层 [...]
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return []


def generate_batch(client: OpenAI, model: str, system_prompt: str,
                   topics: list, batch_size: int, temperature: float,
                   max_retries: int = 3) -> list:
    """调用 API 生成一批对话"""
    user_prompt = build_user_prompt(topics, batch_size)

    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=4096,
            )

            # 尝试使用 json_object 模式
            if attempt == 0:
                try:
                    kwargs["response_format"] = {"type": "json_object"}
                    response = client.chat.completions.create(**kwargs)
                except Exception:
                    # 不支持 json_object，降级
                    kwargs.pop("response_format", None)
                    response = client.chat.completions.create(**kwargs)
            else:
                kwargs.pop("response_format", None)
                response = client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content
            conversations = parse_json_response(content)

            if conversations:
                return conversations

            if attempt < max_retries - 1:
                print(f"    JSON 解析失败，重试中...")

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    API 错误: {e}，{wait}s 后重试...")
                time.sleep(wait)
            else:
                print(f"    API 错误: {e}，跳过此批次")

    return []


# ============================================================
# 阶段5：校验与输出
# ============================================================

def validate_and_fix(conversations: list) -> list:
    """校验并修复生成的对话"""
    valid = []
    for conv in conversations:
        if not isinstance(conv, dict) or "conversations" not in conv:
            continue
        msgs = conv["conversations"]
        if not isinstance(msgs, list) or len(msgs) < 2:
            continue

        # 检查每条消息格式
        ok = True
        for msg in msgs:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                ok = False
                break
            if msg["role"] not in ("user", "assistant"):
                ok = False
                break
            if not isinstance(msg["content"], str) or not msg["content"].strip():
                ok = False
                break
        if not ok:
            continue

        # 确保以 user 开头
        if msgs[0]["role"] != "user":
            msgs = msgs[1:]
        if len(msgs) < 2:
            continue

        # 确保 user/assistant 交替
        fixed = [msgs[0]]
        for msg in msgs[1:]:
            if msg["role"] != fixed[-1]["role"]:
                fixed.append(msg)
            # 连续同 role 则跳过

        # 确保以 assistant 结尾
        if fixed[-1]["role"] == "user":
            fixed = fixed[:-1]
        if len(fixed) < 2:
            continue

        valid.append({"conversations": fixed})

    return valid


def main():
    parser = argparse.ArgumentParser(
        description="数据增强：分析对话风格并生成新对话"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/train_data.json",
        help="输入训练数据路径 (默认 data/train_data.json)"
    )
    parser.add_argument(
        "--base-url", "-u",
        type=str,
        default=None,
        help="API base_url (如 https://api.deepseek.com/v1，也读取 .env OPENAI_BASE_URL)"
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="API key (也读取 .env OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="模型名称 (如 deepseek-chat，也读取 .env AUGMENT_MODEL)"
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help=".env 文件路径 (默认自动查找项目根目录)"
    )
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=20,
        help="生成对话数量 (默认20)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出路径 (默认 {input}_augmented.json)"
    )
    parser.add_argument(
        "--batch-size", "-B",
        type=int,
        default=5,
        help="每批生成数量 (默认5)"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.9,
        help="采样温度 (默认0.9)"
    )
    parser.add_argument(
        "--save-analysis",
        action="store_true",
        help="保存风格分析报告和生成 prompt"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="配置文件路径 (默认自动查找 data/config.json)"
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="不合并原始数据，只输出生成的数据"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="API 错误最大重试次数 (默认3)"
    )

    args = parser.parse_args()

    # 加载 .env（CLI 参数 > .env > 已有环境变量）
    load_dotenv(args.env)

    # 随机种子
    if args.seed is not None:
        random.seed(args.seed)

    # 解析配置：CLI 参数优先，其次 .env/环境变量
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("错误：请通过 -k 参数或 .env OPENAI_API_KEY 提供 API key")
        sys.exit(1)

    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        print("错误：请通过 -u 参数或 .env OPENAI_BASE_URL 提供 API base_url")
        sys.exit(1)

    model = args.model or os.environ.get("AUGMENT_MODEL")
    if not model:
        print("错误：请通过 -m 参数或 .env AUGMENT_MODEL 提供模型名称")
        sys.exit(1)

    # 输出路径
    if args.output:
        output_path = args.output
    else:
        input_p = Path(args.input)
        output_path = str(input_p.parent / f"{input_p.stem}_augmented{input_p.suffix}")

    # 加载配置文件（人设信息）
    persona = None
    config_candidates = [args.config] if args.config else [
        "data/config.json",
        str(Path(__file__).resolve().parent.parent / "data" / "config.json"),
    ]
    for cp in config_candidates:
        if cp and Path(cp).is_file():
            with open(cp, "r", encoding="utf-8") as f:
                chat_config = json.load(f)
            persona = chat_config.get("system_prompt", "")
            if persona:
                print(f"已加载人设配置: {cp}")
            break
    if not persona:
        print("未找到 config.json 或 system_prompt 为空，将仅使用风格分析结果生成")

    # ---- 阶段1：加载 ----
    print("=" * 50)
    print("阶段 1/5：加载与校验输入数据")
    print("=" * 50)
    data = load_and_validate(args.input)

    # ---- 阶段2：风格分析 ----
    print("=" * 50)
    print("阶段 2/5：风格分析")
    print("=" * 50)
    analysis = analyze_style(data)

    # ---- 阶段3：构建 Prompt ----
    print("=" * 50)
    print("阶段 3/5：构建生成 Prompt")
    print("=" * 50)
    sample_conversations = random.sample(data, min(3, len(data)))
    system_prompt = build_system_prompt(analysis, sample_conversations, persona=persona)
    print(f"  System prompt 长度: {len(system_prompt)} 字符")

    # 保存分析报告
    if args.save_analysis:
        output_dir = Path(output_path).parent
        analysis_path = output_dir / "style_analysis.json"
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        print(f"  风格分析已保存: {analysis_path}")

        prompt_path = output_dir / "generation_prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(system_prompt)
        print(f"  生成 prompt 已保存: {prompt_path}")

    # 准备话题
    available_topics = list(TOPIC_POOL)
    random.shuffle(available_topics)

    # ---- 阶段4：API 生成 ----
    print()
    print("=" * 50)
    print("阶段 4/5：调用 API 生成对话")
    print("=" * 50)

    client = OpenAI(base_url=base_url, api_key=api_key)

    all_generated = []
    remaining = args.num
    batch_idx = 0
    total_batches = (args.num + args.batch_size - 1) // args.batch_size

    while remaining > 0:
        batch_size = min(args.batch_size, remaining)
        batch_idx += 1

        # 选取话题
        start = ((batch_idx - 1) * batch_size) % len(available_topics)
        topics = []
        for j in range(batch_size):
            topics.append(available_topics[(start + j) % len(available_topics)])

        print(f"  [{batch_idx}/{total_batches}] 生成 {batch_size} 个对话 "
              f"(已完成 {len(all_generated)}/{args.num})...")

        conversations = generate_batch(
            client, model, system_prompt,
            topics, batch_size, args.temperature,
            max_retries=args.max_retries,
        )

        validated = validate_and_fix(conversations)
        all_generated.extend(validated)
        remaining -= batch_size

        if validated:
            print(f"    成功生成 {len(validated)} 个有效对话")
        else:
            print(f"    本批次未生成有效对话")

    print(f"\n  共生成 {len(all_generated)} 个有效对话（目标 {args.num}）")

    if not all_generated:
        print("错误：未生成任何有效对话，请检查 API 配置")
        sys.exit(1)

    # ---- 阶段5：输出 ----
    print()
    print("=" * 50)
    print("阶段 5/5：校验与输出")
    print("=" * 50)

    if args.no_merge:
        output_data = all_generated
        print(f"  输出生成数据: {len(all_generated)} 组对话")
    else:
        output_data = data + all_generated
        print(f"  合并数据: {len(data)} (原始) + {len(all_generated)} (生成) = {len(output_data)} 组对话")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"  已保存: {output_path}")
    print("\n完成！")


if __name__ == "__main__":
    main()
