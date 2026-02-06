#!/usr/bin/env python3
"""
模型转换脚本：将HuggingFace模型转换为GGUF格式
用于llama.cpp推理

注意：此脚本需要安装llama.cpp并编译convert脚本
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_llama_cpp():
    """检查llama.cpp是否可用"""
    try:
        result = subprocess.run(
            ["python", "-m", "llama_cpp", "--version"],
            capture_output=True,
            text=True
        )
        return True
    except Exception:
        return False


def convert_to_gguf(
    model_path: str,
    output_path: str,
    quantize: str = "q4_k_m"
):
    """
    转换模型为GGUF格式

    Args:
        model_path: HuggingFace模型路径
        output_path: 输出GGUF文件路径
        quantize: 量化方法 (q4_0, q4_k_m, q8_0, f16等)
    """
    print(f"转换模型: {model_path}")
    print(f"输出路径: {output_path}")
    print(f"量化方法: {quantize}")

    # 方法1：使用llama.cpp的convert脚本
    # 需要克隆llama.cpp仓库
    llama_cpp_path = Path.home() / "llama.cpp"
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"

    if convert_script.exists():
        print("\n使用llama.cpp转换脚本...")

        # 先转换为f16
        f16_path = output_path.replace(".gguf", "-f16.gguf")

        cmd = [
            sys.executable,
            str(convert_script),
            model_path,
            "--outfile", f16_path,
            "--outtype", "f16"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"转换失败: {result.stderr}")
            return False

        # 量化
        if quantize != "f16":
            quantize_bin = llama_cpp_path / "build" / "bin" / "llama-quantize"
            if quantize_bin.exists():
                cmd = [str(quantize_bin), f16_path, output_path, quantize]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"量化失败: {result.stderr}")
                    return False

                # 删除f16中间文件
                Path(f16_path).unlink()
            else:
                print("警告：未找到量化工具，保留f16格式")
                Path(f16_path).rename(output_path)

        print(f"\n转换完成: {output_path}")
        return True

    # 方法2：提示用户手动操作
    print("\n" + "="*50)
    print("自动转换不可用，请手动执行以下步骤：")
    print("="*50)
    print("""
1. 安装llama.cpp:
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make

2. 转换模型:
   python convert_hf_to_gguf.py {model_path} --outfile model-f16.gguf --outtype f16

3. 量化模型 (可选，推荐q4_k_m):
   ./llama-quantize model-f16.gguf model-q4_k_m.gguf q4_k_m

量化选项说明:
- q4_0: 最小体积，质量一般
- q4_k_m: 推荐，体积小且质量好
- q8_0: 质量好，体积较大
- f16: 无损，体积最大
""".format(model_path=model_path))

    return False


def main():
    parser = argparse.ArgumentParser(description="转换模型为GGUF格式")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="HuggingFace模型路径（本地或远程）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="model.gguf",
        help="输出GGUF文件路径"
    )
    parser.add_argument(
        "--quantize", "-q",
        type=str,
        default="q4_k_m",
        choices=["q4_0", "q4_k_m", "q5_k_m", "q8_0", "f16"],
        help="量化方法"
    )

    args = parser.parse_args()

    convert_to_gguf(args.model, args.output, args.quantize)


if __name__ == "__main__":
    main()
