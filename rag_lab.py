"""
Local RAG Lab Benchmark Script
Benchmarks chunking strategies for the GitLab Handbook using LlamaIndex, FastEmbed, Postgres+pgvector, and RAGAS.

Requires: Postgres+pgvector with index already built; OPENAI_API_KEY (or set LlamaIndex Settings.llm)
for the query engine to generate answers. Run from project root: python rag_lab.py

Env: EVAL_TOP_K (default 3) = number of chunks retrieved per question; HANDBOOK_DIR for build_index.
"""
import json
import logging
import os
from pathlib import Path

from rich.table import Table
from rich.console import Console
from llama_index.embeddings.fastembed import FastEmbedEmbedding
# from llama_index.embeddings.ollama import OllamaEmbedding

from raglab.markdown_loader import MarkdownLoader
from raglab.chunking_strategy_factory import ChunkingStrategyFactory
from raglab.pgvector_store_factory import PGVectorStoreFactory
from raglab.index_builder import IndexBuilder
from raglab.ragas_evaluator import RAGASEvaluator
from raglab.utils import write_evaluation_report_md, write_combined_report_md

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LocalRAGLab")

EVAL_QUESTIONS_PATH = Path(__file__).resolve().parent / "eval_questions.json"
EVAL_TOP_K = int(os.getenv("EVAL_TOP_K", "3"))


def load_eval_questions(path: Path | None = None) -> list:
    """Load question/reference pairs from JSON config."""
    path = path or EVAL_QUESTIONS_PATH
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not data or not all(isinstance(q, dict) and "question" in q and "reference" in q for q in data):
        raise ValueError(f"{path} must be a JSON array of objects with 'question' and 'reference' keys.")
    return data


class RAGLabBenchmark:
    def __init__(self, handbook_dir: str, questions_path: Path | None = None):
        self.handbook_dir = handbook_dir
        self.loader = MarkdownLoader(handbook_dir)
        self.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
        # self.embed_model = OllamaEmbedding(model_name="your-ollama-model")
        self.strategies = ChunkingStrategyFactory.get_strategies(self.embed_model)
        self.questions = load_eval_questions(questions_path)

    def run(self):
        results = []
        for name, parser in self.strategies.items():
            logger.info(f"Evaluating strategy: {name}")
            table_name = f"handbook_chunks_{name.lower().replace(' ', '_')}"
            vector_store = PGVectorStoreFactory.create(table_name=table_name)
            # Load index from existing vector store (do not rebuild)
            index = IndexBuilder.load(parser, self.embed_model, vector_store)
            faithfulness, precision, _, dataFrame = RAGASEvaluator.evaluate(
                index, self.questions, top_k=EVAL_TOP_K
            )
            results.append((name, faithfulness, precision, dataFrame))

            write_evaluation_report_md(
                dataFrame,
                f"ragas_evaluation_report_{name.lower().replace(' ', '_')}.md",
                strategy_name=name,
                run_metadata={"top_k": EVAL_TOP_K, "n_questions": len(self.questions)},
            )

        write_combined_report_md(
            results,
            "ragas_evaluation_combined.md",
            run_metadata={"top_k": EVAL_TOP_K, "n_questions": len(self.questions)},
        )
        self.display_results(results)

    @staticmethod
    def display_results(results):
        console = Console()
        table = Table(title="RAG Chunking Strategy Benchmark", show_lines=True)
        table.add_column("Strategy", style="bold magenta")
        table.add_column("Faithfulness", style="green")
        table.add_column("Context Precision", style="cyan")
        for name, faithfulness, precision, _ in results:
            table.add_row(
                name,
                f"{faithfulness:.3f}",
                f"{precision:.3f}",
            )
        console.print(table)

def main():
    handbook_dir = "../handbook/content/handbook"
    benchmark = RAGLabBenchmark(handbook_dir)
    benchmark.run()

if __name__ == "__main__":
    main()
