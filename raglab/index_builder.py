from llama_index.core import VectorStoreIndex, Settings, Document
from typing import List

class IndexBuilder:
    @staticmethod
    def build(docs: List[Document], node_parser, embed_model, vector_store) -> VectorStoreIndex:
        Settings.embed_model = embed_model
        Settings.node_parser = node_parser
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model, settings=Settings, show_progress=True    
        )
        nodes = node_parser.get_nodes_from_documents(docs)
        index.insert_nodes(nodes, show_progress=True)
        return index

    @staticmethod
    async def build_async(docs: List[Document], node_parser, embed_model, vector_store) -> VectorStoreIndex:
        """Build index using async node parsing. Use this for parsers that require async (e.g. MarkdownElementNodeParser) to avoid nested event loops."""
        Settings.embed_model = embed_model
        Settings.node_parser = node_parser
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model, settings=Settings, show_progress=True
        )
        nodes = await node_parser.aget_nodes_from_documents(docs, show_progress=True)
        index.insert_nodes(nodes, show_progress=True)
        return index

    @staticmethod
    def load(node_parser, embed_model, vector_store) -> VectorStoreIndex:
        Settings.embed_model = embed_model
        Settings.node_parser = node_parser
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model, settings=Settings
        )
        return index
