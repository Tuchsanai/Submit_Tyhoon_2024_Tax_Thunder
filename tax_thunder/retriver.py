import pickle


import os
import numpy as np
from mynode import MyNode


os.environ["HF_HUB_CACHE"] = "./cache"


### Load tree
with open("./summary_tree.pkl", "rb") as file:
    loaded_data = pickle.load(file)

importer = DictImporter()
root = importer.import_(loaded_data)

### Query function
model = 
reranker = 


def compute_top_k_nodes(
    query_node: MyNode, docs_nodes: list[MyNode], top_k: int = 3
) -> list[tuple[MyNode, float]]:
   









    return [(top_k_nodes[i], rerank_scores[i]) for i in reranked_indices]


def get_query_node(query: str) -> MyNode:
    emb = model.encode(
        query, return_dense=True, return_sparse=True, return_colbert_vecs=False
    )
    return MyNode(
        text=query,
        dense_embedding=emb["dense_vecs"],
        sparse_embedding=emb["lexical_weights"],
    )


def query_pipe(query: str) -> list[MyNode]:
    query_node = get_query_node(query)

   

    return result_nodes


def format_docs(doc_nodes: list[MyNode]):
    doc_with_topic = [
        node.parent.text.split("\n")[0] + "\n" + node.text for node in doc_nodes
    ]
    return doc_with_topic


def get_retriver_chain():
    prompt = """\
    คุณเป็นผู้ช่วยในงานตอบคำถาม ใช้ข้อมูลที่ดึงมาต่อไปนี้เพื่อตอบคำถาม ถ้าไม่รู้คำตอบก็บอกว่าไม่รู้ ตอบไม่เกินสามประโยคและตอบคำถามให้กระชับ

    คำถาม: {question} 
    ข้อมูลที่ดึงมา: {context} 
    คำตอบ:
    """

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. You always answer in Thai."),
            ("human", prompt),
        ]
    )

    gen_chain = final_prompt | ChatOpenAI(
        api_key="xxxxxxx",
        openai_api_base="https://api.opentyphoon.ai/v1",
        model="typhoon-v1.5x-70b-instruct",
        temperature=0,
    )
    return gen_chain


retriver_chain = get_retriver_chain()
