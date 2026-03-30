from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy import create_engine, inspect
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IndexBuilder")


def write_evaluation_report_md(
    df: pd.DataFrame,
    path: str,
    strategy_name: Optional[str] = None,
    include_scores: bool = True,
    include_reference: bool = True,
    include_context: bool = True,
    run_metadata: Optional[dict] = None,
) -> None:
    """
    Write a RAG evaluation report. Order: Question → Answer → Reference → Scores → Context (collapsible).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sections: List[str] = []
    if strategy_name:
        sections.append(f"# RAG Evaluation: {strategy_name}\n\n")
    if run_metadata:
        parts = [f"- **{k}**: {v}" for k, v in run_metadata.items()]
        sections.append("\n".join(parts) + "\n\n")
    for i, row in df.iterrows():
        num = i + 1
        sections.append(f"## {num}. {row['question']}\n\n")
        sections.append("**Answer:** " + str(row["answer"]).replace("\n", " ") + "\n\n")
        if include_reference and "reference" in row:
            sections.append("**Reference:** " + str(row["reference"]).replace("\n", " ") + "\n\n")
        if include_scores:
            parts = []
            if "faithfulness_adjusted" in row and pd.notna(row.get("faithfulness_adjusted")):
                parts.append(f"Faithfulness: {row['faithfulness_adjusted']:.3f}")
            if "context_precision_adjusted" in row and pd.notna(row.get("context_precision_adjusted")):
                parts.append(f"Context precision: {row['context_precision_adjusted']:.3f}")
            if row.get("abstained"):
                parts.append("*(abstained)*")
            if parts:
                sections.append("**Scores:** " + " | ".join(parts) + "\n\n")
        if include_context:
            contexts = row.get("contexts") or []
            if isinstance(contexts, str):
                contexts = [contexts]
            if contexts:
                sections.append("<details><summary>Retrieved context</summary>\n\n")
                for j, ctx in enumerate(contexts):
                    if len(contexts) > 1:
                        sections.append(f"*Chunk {j + 1}:*\n\n")
                    sections.append("```\n")
                    sections.append(ctx[:6000] + ("..." if len(ctx) > 6000 else ""))
                    sections.append("\n```\n\n")
                sections.append("</details>\n\n")
        sections.append("---\n\n")
    path.write_text("".join(sections), encoding="utf-8")
    logger.info("Wrote evaluation report: %s", path)


def write_combined_report_md(
    results: List[tuple],
    path: str,
    run_metadata: Optional[dict] = None,
) -> None:
    """
    Write one combined report with a single summary table (all strategies) and per-question breakdown.
    results: list of (strategy_name, avg_faithfulness, avg_precision, dataFrame).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sections: List[str] = []

    sections.append("# RAG Chunking Benchmark — Summary\n\n")
    if run_metadata:
        parts = [f"- **{k}**: {v}" for k, v in run_metadata.items()]
        sections.append("\n".join(parts) + "\n\n")

    sections.append("## Overall results\n\n")
    sections.append("| Strategy | Faithfulness | Context precision |\n")
    sections.append("|----------|-------------:|-------------------:|\n")
    for name, faith, prec, _ in results:
        sections.append(f"| {name} | {faith:.3f} | {prec:.3f} |\n")
    sections.append("\n")
    sections.append(
        "*Fixed-size chunking cuts at arbitrary token boundaries and often splits sentences or paragraphs. "
        "Semantic chunking (see [LlamaIndex semantic chunking](https://developers.llamaindex.ai/python/examples/node_parsers/semantic_chunking/)) "
        "picks breakpoints by embedding similarity so each chunk stays semantically coherent—typically better than size-based or even sentence-based chunking when answers span multiple sentences.*\n\n"
    )

    if not results:
        path.write_text("".join(sections), encoding="utf-8")
        logger.info("Wrote combined report: %s", path)
        return

    sections.append("## Per-question scores (Faithfulness / Context precision)\n\n")
    n_questions = len(results[0][3])
    for q_idx in range(n_questions):
        row = results[0][3].iloc[q_idx]
        q_short = (str(row["question"])[:70] + "…") if len(str(row["question"])) > 70 else str(row["question"])
        sections.append(f"### Q{q_idx + 1}: {q_short}\n\n")
        sections.append("| Strategy | Faithfulness | Context precision |\n")
        sections.append("|----------|-------------:|-------------------:|\n")
        for name, _, _, df in results:
            r = df.iloc[q_idx]
            f = r.get("faithfulness_adjusted", 0)
            p = r.get("context_precision_adjusted", 0)
            f = float(f) if pd.notna(f) else 0.0
            p = float(p) if pd.notna(p) else 0.0
            sections.append(f"| {name} | {f:.3f} | {p:.3f} |\n")
        sections.append("\n")

    path.write_text("".join(sections), encoding="utf-8")
    logger.info("Wrote combined report: %s", path)


def table_exists(table_name: str, db_url: str) -> bool:
    """
    Check if a table exists in the given Postgres database.
    Args:
        table_name (str): Name of the table to check.
        db_url (str): SQLAlchemy database URL.
    Returns:
        bool: True if table exists, False otherwise.
    """
    table_name_with_prefix = f"data_{table_name}"
    engine = create_engine(db_url)
    inspector = inspect(engine)
    logger.info(f"Checking if table '{table_name_with_prefix}' exists in the database.")
    logger.info(f"inspector.get_table_names(): {inspector.get_table_names()}")

    return inspector.has_table(table_name_with_prefix)
