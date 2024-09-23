#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :retrivel_industry.py
# @Time      :2024/09/22 18:28:30
# @Author    :Lifeng
# @Description :
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
import faiss
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
import os
import json
from langchain_openai import ChatOpenAI
import os
from tqdm import tqdm
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from BCErerank import BCERerank
# from BCEmbedding.tools.langchain import BCERerank
from langchain.retrievers import ContextualCompressionRetriever

os.environ["OPENAI_API_KEY"] = "None"
emb_model = "/root/autodl-tmp/Model/bce-embedding-base_v1"
embedding_model_kwargs = {'device': 'cuda:0'}
embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
  model_name=emb_model,
  model_kwargs=embedding_model_kwargs,
  encode_kwargs=embedding_encode_kwargs
)

reranker_args = {'model': '/root/autodl-tmp/Model/bce-reranker-base_v1', 'top_n': 3, 'device': 'cuda:0'}
reranker_model = BCERerank(**reranker_args)

vector_store = FAISS.load_local(
    "faiss_industry", embeddings, allow_dangerous_deserialization=True
)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)

file_cache_path = "/root/autodl-tmp/Code/rag_demo/cache"
local_file_store = create_kv_docstore(LocalFileStore(file_cache_path))
retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=local_file_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={
        "k": 10
    }
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker_model, base_retriever=retriever
)

def format_docs(docs):
    return "\n".join(
        [f"参考内容{i+1}\n{doc.page_content}\n" for i, doc in enumerate(docs)]
    )

llm = ChatOpenAI(model_name="qwen2-7B",
                 base_url= "http://localhost:7000/v1",
                temperature=0,
                max_tokens=300)
template = """你是一名根据参考内容回答用户问题的机器人，你的职责是：根据提供的参考内容回答用户的问题。如果参考内容与问题不相关，你可以选择忽略参考内容，只回答问题。

##参考内容：
{context}

##用户问题：
{question}

请根据已知内容，一步一步回答问题。
"""

file_path = "/root/autodl-tmp/Code/LLaMA-Factory/data/QA_test.json"
# file_path = "/root/autodl-tmp/Code/LLaMA-Factory/data/QA_merged_clean_update.json"
query_list = []
label_list = []
data = []
with open(file_path, 'r', encoding="utf-8") as f:
    data = json.load(f)

for it in data:
    query_list.append(it["instruction"])
    label_list.append(it["output"])

pred_list = []
for i, query in tqdm(enumerate(query_list), total=len(query_list), desc="Processing"):

    refs = compression_retriever.get_relevant_documents(query)
    context = format_docs(refs)
    prompt = template.format(context=context, question=query)
    result = llm.invoke(prompt)
    pred = result.content
    with open("out2.log", "a", encoding="utf-8") as f:
        f.write(prompt)
        f.write("======================================\n")
        f.write(pred + "\n")
        f.write("======================================\n")
    pred_list.append(pred)
    with open("./llm_ans_temp_0_max_chunk600_rerank.jsonl", "a", encoding="utf-8") as f:
        res = {"instruction": query_list[i], "label": label_list[i], "pred": pred}
        f.write(json.dumps(res, ensure_ascii=False) + "\n")

