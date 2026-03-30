"""
Sliding-window node parser for RAG indexing.
Produces overlapping chunks by sliding a fixed-size window with a given stride,
so the same content can appear in multiple chunks (unlike fixed-size with a single overlap at boundaries).
"""
from typing import List, Sequence

from llama_index.core.schema import BaseNode, Document, TextNode


class SlidingWindowNodeParser:
    """
    Split documents into overlapping chunks using a sliding window.
    Same as fixed window size, but stride < window_size so chunks overlap heavily
    and the same span of text appears in several chunks (better recall at boundary regions).
    """

    def __init__(
        self,
        window_size: int = 512,
        stride: int = 128,
        include_metadata: bool = True,
    ):
        if stride <= 0 or stride > window_size:
            raise ValueError("stride must be in (0, window_size] for overlapping windows.")
        self.window_size = window_size
        self.stride = stride
        self.include_metadata = include_metadata

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs,
    ) -> List[BaseNode]:
        nodes: List[BaseNode] = []
        for doc in documents:
            text = doc.get_content()
            metadata = dict(doc.metadata) if self.include_metadata else {}
            start = 0
            while start < len(text):
                end = min(start + self.window_size, len(text))
                chunk = text[start:end]
                if chunk.strip():
                    nodes.append(TextNode(text=chunk, metadata={**metadata}))
                start += self.stride
                if start >= len(text):
                    break
        return nodes
