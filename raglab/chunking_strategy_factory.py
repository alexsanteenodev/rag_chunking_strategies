CHUNK_SIZE = 128
CHUNK_OVERLAP = 0

from llama_index.core.node_parser import (
    MarkdownNodeParser,
    TokenTextSplitter,
    SentenceSplitter,
    SemanticSplitterNodeParser,
)


class ChunkingStrategyFactory:
    @staticmethod
    def get_strategies(embed_model):
        """Return chunking strategies. Markdown Syntax-Aware uses MarkdownNodeParser (no LLM)."""
        return {
            "Fixed-Size": TokenTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            ),
            "Recursive Character": SentenceSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            ),
            # Match LlamaIndex example: https://developers.llamaindex.ai/python/examples/node_parsers/semantic_chunking/
            "Semantic Chunking": SemanticSplitterNodeParser(
                embed_model=embed_model,
                buffer_size=1,
                breakpoint_percentile_threshold=95,
            ),
            "Markdown Syntax-Aware": MarkdownNodeParser(),
        }
