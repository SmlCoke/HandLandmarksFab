#!/usr/bin/env python3
"""
图片降采样工具
从输入文件夹中每 N 张图片取 1 张，复制到输出文件夹。
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# 支持的图片扩展名（不区分大小写）
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}


def is_image_file(filename: str) -> bool:
    """判断文件是否为常见图片格式"""
    ext = Path(filename).suffix.lower()
    return ext in IMAGE_EXTENSIONS


def get_image_files(input_dir: str):
    """获取输入目录下所有图片文件，并按文件名排序"""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"输入目录不存在或不是目录: {input_dir}")

    # 收集所有图片文件（仅文件，忽略子目录）
    image_files = []
    for entry in input_path.iterdir():
        if entry.is_file() and is_image_file(entry.name):
            image_files.append(entry)

    # 按文件名排序（自然顺序：按字符串排序，数字会被正确处理）
    image_files.sort(key=lambda p: p.name)
    return image_files


def downsample_images(input_dir: str, output_dir: str, n: int):
    """
    执行降采样复制
    :param input_dir:  输入文件夹
    :param output_dir: 输出文件夹
    :param n:          降采样因子（每 N 张取 1 张）
    """
    if n < 1:
        raise ValueError("降采样因子 N 必须为正整数（N >= 1）")

    # 获取所有图片文件
    files = get_image_files(input_dir)
    total = len(files)
    if total == 0:
        print("警告：输入目录中未找到任何图片文件。")
        return

    # 计算需要取出的索引
    indices = range(0, total, n)   # 从第 0 张开始，每隔 N 张取 1 张
    selected_count = len(list(indices))  # 注意：range 是迭代器，但用 len 会消耗，需重新创建

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"共找到 {total} 张图片，降采样因子 N={n}，将复制 {selected_count} 张到 '{output_dir}'")

    # 执行复制
    copied = 0
    for i in range(0, total, n):
        src = files[i]
        dst = output_path / src.name
        try:
            shutil.copy2(src, dst)   # copy2 保留元数据
            copied += 1
        except Exception as e:
            print(f"复制文件 {src.name} 失败: {e}")

    print(f"完成！成功复制 {copied} 张图片。")


def main():
    parser = argparse.ArgumentParser(
        description="对连续拍摄的图片进行降采样（每 N 张取 1 张），复制到输出文件夹。"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="输入文件夹路径（包含图片文件）"
    )
    parser.add_argument(
        "N",
        type=int,
        help="降采样因子，正整数（如 2 表示每 2 张取 1 张）"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="输出文件夹路径（不存在时会自动创建）"
    )

    args = parser.parse_args()

    try:
        downsample_images(args.input_dir, args.output_dir, args.N)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()