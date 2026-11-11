# -*- coding: utf-8 -*-
"""L2-02 文档加载与分块：对 AI_GUIDE_ALL.md 做递归分块并打印统计"""
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

PATH = "../../AI_GUIDE_ALL.md"

def main():
    with open(PATH, encoding="utf-8") as f:
        text = f.read()
    print("原文长度(字符):", len(text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", " "],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    print("chunk 数:", len(chunks))
    lens = [len(c) for c in chunks]
    print("平均长度:", round(sum(lens) / len(lens), 1))
    print("最短/最长:", min(lens), max(lens))
    print("\n--- 第 1 个 chunk ---")
    print(chunks[0][:400])
    print("\n--- 第 2 个 chunk ---")
    print(chunks[1][:400])

if __name__ == "__main__":
    main()
