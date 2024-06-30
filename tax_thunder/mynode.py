from anytree import NodeMixin, RenderTree, Node
import numpy as np


class MyNode(NodeMixin):
    def __init__(
        self,
        text: str,
        dense_embedding: np.array = None,
        sparse_embedding: np.array = None,
        parent: Node = None,
    ):
        self.text = text
        self.dense_embedding = dense_embedding
        self.sparse_embedding = sparse_embedding
        self.parent = parent
