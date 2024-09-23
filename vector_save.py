#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :rag_test.py
# @Time      :2024/09/22 16:53:18
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
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
import json
from langchain_openai import ChatOpenAI
import os
from tqdm import tqdm
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore

os.environ["OPENAI_API_KEY"] = "None"
emb_model = "/root/autodl-tmp/Model/bce-embedding-base_v1"
embedding_model_kwargs = {'device': 'cuda:0'}
embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
  model_name=emb_model,
  model_kwargs=embedding_model_kwargs,
  encode_kwargs=embedding_encode_kwargs
)

# model_name = "/root/autodl-tmp/Model/bge-large-en-v1.5"
# model_kwargs = {'device': 'cuda'}
# encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
# embeddings = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs,
#     query_instruction="为这个句子生成表示以用于检索相关文章："
# )

def get_loaders(dir_path):
    loaders = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path):
            loaders.append(TextLoader(file_path))
    return loaders

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)

vector_store = FAISS(
    embedding_function=embeddings,
    # index=faiss.IndexFlatL2(1024),
    index=faiss.IndexFlatL2(768),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

file_cache_path = "/root/autodl-tmp/Code/rag_demo/cache"
local_file_store = create_kv_docstore(LocalFileStore(file_cache_path))
retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=local_file_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

txt_path = "/root/autodl-tmp/Code/rag_demo/industry_pretrain_data"
loaders = get_loaders(txt_path)
docs = []
for loader in loaders:
    docs.extend(loader.load())
retriever.add_documents(docs)

vector_store.save_local("faiss_industry")
# print("加载完成...")







