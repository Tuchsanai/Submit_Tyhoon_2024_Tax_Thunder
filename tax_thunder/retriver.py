import pickle
from anytree.importer import DictImporter
from FlagEmbedding import BGEM3FlagModel, FlagReranker
import os
import numpy as np
from mynode import MyNode
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

os.environ["HF_HUB_CACHE"] = "./cache"


### Load tree
with open("./summary_tree.pkl", "rb") as file:
    loaded_data = pickle.load(file)

importer = DictImporter()
root = importer.import_(loaded_data)


### Query function
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)


def compute_top_k_nodes(
    query_node: MyNode, docs_nodes: list[MyNode], top_k: int = 3
) -> list[tuple[MyNode, float]]:
    """
    Compute scores using sparse+dense embeddings and re-rank the top results.

    Args:
    query_node: The query node with pre-computed embeddings
    docs_nodes: List of document nodes with pre-computed embeddings
    top_k: Number of top results to return

    Returns:
    List of tuples, each containing a top-k MyNode object and its corresponding score after re-ranking
    """
    scores = []
    for doc_node in docs_nodes:
        similarity = query_node.dense_embedding @ doc_node.dense_embedding.T
        lexical_scores = model.compute_lexical_matching_score(
            query_node.sparse_embedding, doc_node.sparse_embedding
        )
        sparse_dense = (similarity + lexical_scores) / 2
        scores.append(sparse_dense)

    scores = np.array(scores).flatten()
    top_k_indices = np.argsort(scores)[::-1][:top_k]

    top_k_nodes = [docs_nodes[i] for i in top_k_indices]
    top_k_scores = scores[top_k_indices]
    sentence_pairs = [[query_node.text, node.text] for node in top_k_nodes]

    rerank_scores = reranker.compute_score(sentence_pairs)
    if isinstance(rerank_scores, float):
        rerank_scores = [rerank_scores]

    reranked_indices = np.argsort(rerank_scores)[::-1]

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

    result_nodes = []

    # level 1
    best_level1_node_score_pair = compute_top_k_nodes(
        query_node, root.children, top_k=1
    )[0]

    # level 2
    top_level2_node_score_pairs = compute_top_k_nodes(
        query_node, best_level1_node_score_pair[0].children, top_k=3
    )
    for level2_node_score_pair in top_level2_node_score_pairs:
        level2_node, level2_score = level2_node_score_pair

        if len(level2_node.children) == 1:
            result_nodes.append((level2_node.children[0], level2_score))
        else:
            top_level3_node_score_pairs = compute_top_k_nodes(
                query_node, level2_node.children, top_k=3
            )
            for level3_node_score_pair in top_level3_node_score_pairs:
                level3_node, level3_score = level3_node_score_pair

                result_nodes.append((level3_node.children[0], level3_score))

    # sort best score
    result_nodes = sorted(result_nodes, key=lambda x: x[-1], reverse=True)

    # get only node
    result_nodes = [node_score[0] for node_score in result_nodes]

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
        api_key="sk-L1fn4b3Oe2dPbao8JREMWXoAqJcqB7PXcgKaT57G8IUo5uVH",
        openai_api_base="https://api.opentyphoon.ai/v1",
        model="typhoon-v1.5x-70b-instruct",
        temperature=0,
    )
    return gen_chain


retriver_chain = get_retriver_chain()
