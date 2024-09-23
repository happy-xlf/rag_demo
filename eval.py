#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :eval.py
# @Time      :2024/09/22 20:48:40
# @Author    :Lifeng
# @Description :
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
import jieba  # type: ignore
import json
import numpy as np

data = []
# with open("./llm_ans_temp_0_max_chunk600.jsonl", "r", encoding="utf-8") as f:
with open("./llm_ans_temp_0_max_chunk600_rerank.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

decoded_preds = [item["pred"] for item in data]
decoded_labels = [item["label"] for item in data]

score_dict = {"bleu-4": [], "rouge-1": [], "rouge-2": [], "rouge-l": []}
for pred, label in zip(decoded_preds, decoded_labels):
    hypothesis = list(jieba.cut(pred))
    reference = list(jieba.cut(label))

    if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
        result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    else:
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        result = scores[0]

    for k, v in result.items():
        score_dict[k].append(round(v["f"] * 100, 4))

    bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
    score_dict["bleu-4"].append(round(bleu_score * 100, 4))

result = {k: float(np.mean(v)) for k, v in score_dict.items()}
print(result)
# with open("score.json", "w", encoding="utf-8") as f:
#     json.dump(result, f, ensure_ascii=False, indent=4)