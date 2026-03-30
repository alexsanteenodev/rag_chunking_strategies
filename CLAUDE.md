# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

A benchmark comparing 4 document chunking strategies for RAG (Retrieval-Augmented Generation) using the GitLab Handbook as a test corpus. It measures retrieval quality via RAGAS metrics (faithfulness and context precision).

## Commands

### Setup
```bash
docker compose up -d                          # Start Postgres+pgvector
python3 -m venv .venv && source .venv/bin/activate
pip install llama-index rich ragas fastembed
```

### Run
```bash
python build_index.py   # Parse docs, embed, and store in Postgres (skip if tables exist)
python rag_lab.py       # Run RAGAS evaluation across all strategies and generate reports
```

### Environment Variables
- `HANDBOOK_DIR` — path to handbook markdown files (default: `../handbook/content/handbook`)
- `EVAL_TOP_K` — chunks retrieved per query (default: 3)
- `OPENAI_API_KEY` — required at runtime for RAGAS LLM-based scoring

## Architecture

**Two-phase workflow:**

1. **Index phase** (`build_index.py`): Loads `.md` files → parses with each chunking strategy → embeds with FastEmbed (`BAAI/bge-small-en-v1.5`, 384-dim) → stores in separate Postgres tables via pgvector.

2. **Eval phase** (`rag_lab.py`): Loads each index from Postgres → queries with 28 questions from `eval_questions.json` using a custom RAG-only prompt → scores answers with RAGAS → writes markdown reports.

**Core module: `raglab/`**
- `chunking_strategy_factory.py` — Factory returning one of 4 chunking strategies: `fixed_size` (TokenTextSplitter), `recursive_character` (SentenceSplitter), `semantic_chunking` (SemanticSplitterNodeParser, 95th percentile breakpoints), `markdown_syntax_aware` (MarkdownNodeParser)
- `index_builder.py` — Builds or loads a `VectorStoreIndex` from a `PGVectorStore`; `build_async` handles async parsers (semantic chunking)
- `ragas_evaluator.py` — Wraps RAGAS evaluation; enforces RAG-only answers via custom prompt; detects abstentions; returns `(faithfulness, context_precision, relevance, dataframe)`
- `utils.py` — Generates per-strategy and combined comparison markdown reports
- `pgvector_store_factory.py` — Creates `PGVectorStore` with configurable table names
- `markdown_loader.py` — Recursively loads `.md` files as LlamaIndex `Document` objects
- `db_config.py` — Hardcoded Postgres connection settings (`localhost:5432`, db=`raglab`, user=`raguser`, pass=`ragpass`)

**Postgres table naming:** Each strategy gets its own table prefixed with `data_` (e.g., `data_fixed_size`). `utils.table_exists()` checks for existing tables to skip re-indexing.

## No Tests

There are no automated tests. Validation is done by running the full pipeline and inspecting generated reports (`ragas_evaluation_report_*.md`, `ragas_evaluation_combined.md`).
