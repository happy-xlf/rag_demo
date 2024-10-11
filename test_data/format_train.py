#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :format_train.py
# @Time      :2024/09/25 15:32:39
# @Author    :Lifeng
# @Description :
import json

data = []
with open("./rag_qa_test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        tmp = json.loads(line)
        dt = {"instruction": tmp["query"], "input": "", "output": tmp["answer"]}
        data.append(dt)

with open("./rag_qa_test.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)