#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :saple_chat.py
# @Time      :2024/07/19 14:59:44
# @Author    :Lifeng
# @Description :
import sys
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = "None"

def vllm_chat(model_name) -> ChatOpenAI:
    llm = ChatOpenAI(model_name=model_name, base_url= "http://localhost:7000/v1")
    return llm

model = vllm_chat(model_name = "Qwen2-7B")

messages = [
    SystemMessage(content="你是一个聊天机器人，请根据用户输入回答问题"),
    HumanMessage(content="在长时间无操作后，仪器会回到哪个状态？"),
]

res = model.invoke(messages)
print(res.content)
