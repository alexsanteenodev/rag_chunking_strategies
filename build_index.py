"""
Builds vector indexes for all chunking strategies and stores them in Postgres+pgvector.
Run this script only when you want to (re)build the index from your markdown files.
To rebuild an existing table, drop it first (e.g. DROP TABLE handbook_chunks_fixed_size;).

Markdown Syntax-Aware uses MarkdownNodeParser (structure-only, no LLM).
"""
import logging
import os

from llama_index.embeddings.fastembed import FastEmbedEmbedding

from raglab.markdown_loader import MarkdownLoader
from raglab.chunking_strategy_factory import ChunkingStrategyFactory
from raglab.pgvector_store_factory import PGVectorStoreFactory
from raglab.index_builder import IndexBuilder
from raglab.utils import table_exists
from raglab.db_config import DB_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IndexBuilder")

# Use the same handbook root as rag_lab.py so evals run against the content you indexed.
HANDBOOK_DIR = os.getenv("HANDBOOK_DIR", "../handbook/content/handbook")


def main():
    loader = MarkdownLoader(HANDBOOK_DIR)
    docs = loader.load()
    if not docs:
        logger.warning("No markdown files found. Exiting.")
        return
    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    strategies = ChunkingStrategyFactory.get_strategies(embed_model)
    for name, parser in strategies.items():
        logger.info(f"Building index for strategy: {name}")
        table_name = f"handbook_chunks_{name.lower().replace(' ', '_')}"
        if table_exists(table_name, DB_URL):
            logger.info(f"Table {table_name} already exists. Skipping index build.")
            continue

        logger.info(f"Table doesn't exist. Creating table for strategy: {table_name}")
        vector_store = PGVectorStoreFactory.create(table_name=table_name)
        logger.info(f"Vector store initialized for strategy: {name}")
        index = IndexBuilder.build(docs, parser, embed_model, vector_store)
        index.storage_context.persist()  # Persist the index to the vector store
        logger.info(f"Index built and stored in table: {table_name}")


if __name__ == "__main__":
    main()
