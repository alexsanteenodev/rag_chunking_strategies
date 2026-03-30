# Local RAG Lab Benchmark

This project benchmarks three chunking strategies for the GitLab Handbook using LlamaIndex, FastEmbed, Postgres+pgvector, and RAGAS.

## Structure

- `raglab/` — Core modules (each class in its own file)
- `rag_lab.py` — Main script
- `docker-compose.yml` — Starts Postgres+pgvector

## Setup & Run

1. **Start the database:**
   ```bash
   docker compose up -d
   ```

2. **Create and activate a Python virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install llama-index rich ragas fastembed
   ```

4. **Run the benchmark:**
   ```bash
   python rag_lab.py
   ```

## Quick Start with Claude Code

To let Claude set everything up and run the full benchmark automatically:

```
docker compose up -d && python build_index.py && python rag_lab.py
```

Or tell Claude: *"Start Docker, build the index, and run the evaluation"* and it will run the above.

## Notes

- Place your Markdown files in `handbook/content/handbook/`.
- Adjust the embedding model in `rag_lab.py` if you want to use Ollama.
- Results are printed in a rich terminal table comparing Faithfulness and Context Precision for all strategies.
