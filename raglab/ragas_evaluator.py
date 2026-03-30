import asyncio
import logging
import math
from typing import List, Tuple

import pandas as pd
from datasets import Dataset
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings
from llama_index.core import VectorStoreIndex, PromptTemplate
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision

logger = logging.getLogger("LocalRAGLab")

# RAGAS EmbeddingUsageEvent expects model to be a string; FastEmbedEmbeddings.model is the object.
# Wrap so getattr(..., "model", None) returns the model name string and tracking does not raise.
class _FastEmbedForRagas(Embeddings):
    """Thin wrapper so RAGAS analytics get model name (string) instead of the embedding object."""
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._embeddings = FastEmbedEmbeddings(model_name=model_name)
        self.model = model_name  # RAGAS LangchainEmbeddingsWrapper uses this for EmbeddingUsageEvent

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embeddings.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embeddings.embed_documents, texts)


# LangChain embeddings for RAGAS (must have embed_query / embed_documents).
RAGAS_EMBEDDINGS = _FastEmbedForRagas(model_name="BAAI/bge-small-en-v1.5")

# Phrases that indicate the model abstained (no substantive answer)
ABSTAIN_PHRASES = (
    "i don't know",
    "i don't have enough information to answer this question.",
    "i don't have enough information to answer this question",
)

def _is_abstaining(answer: str) -> bool:
    """True if answer is empty or an explicit abstention (should score 0 faithfulness)."""
    if not answer or not answer.strip():
        return True
    lower = answer.strip().lower()
    if lower in ABSTAIN_PHRASES:
        return True
    if "don't have enough information" in lower:
        return True
    return False

def _safe_score(value, default: float = 0.0) -> float:
    """Coerce NaN/None to default so averages are well-defined."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        f = float(value)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default

class RAGASEvaluator:
    @staticmethod
    def evaluate(index: VectorStoreIndex, questions: List[dict], top_k: int = 3) -> Tuple[float, float, float, pd.DataFrame]:
        data = []
        # Custom prompt to enforce RAG-only answers
        text_qa_template = PromptTemplate(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the question: {query_str}\n"
            "If the context does not contain the answer, say 'I don't have enough information to answer this question.'\n"
        )
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            text_qa_template=text_qa_template
        )
        for q_item in questions:
            q = q_item["question"]
            try:
                response = query_engine.query(q)
                # Extract answer and contexts from response
                answer = str(response)
                logger.debug("Q: %s -> %s", q[:60], answer[:80] + "..." if len(answer) > 80 else answer)
                contexts = []
                if hasattr(response, 'source_nodes'):
                    contexts = [str(node.text) for node in response.source_nodes]
                data.append({
                    "question": q,
                    "answer": answer,
                    "contexts": contexts,
                    "reference": q_item['reference']
                })
            except Exception as e:
                logger.error(f"Evaluation failed for question '{q}': {e}")
        if not data:
            return 0.0, 0.0, 0.0, pd.DataFrame()
        dataFrame = pd.DataFrame(data)

        hf_dataset = Dataset.from_pandas(dataFrame)
        result = evaluate(
            hf_dataset,
            metrics=[faithfulness, context_precision],
            embeddings=RAGAS_EMBEDDINGS,
        )
        # result.scores is a list of dicts, one per sample, without answer_relevancy now
        answers = dataFrame["answer"].tolist()
        faithfulness_scores = []
        context_precision_scores = []
        relevance_scores = []
        for i, row in enumerate(result.scores):
            answer_raw = answers[i]
            is_abst = _is_abstaining(answer_raw)
            # Abstaining = no claim from context → 0.0 faithfulness (and 0.0 context precision for strict benchmarking)
            raw_faith = _safe_score(row.get("faithfulness"))
            raw_prec = _safe_score(row.get("context_precision"))
            raw_relevance = 0.0
            if is_abst:
                faithfulness_scores.append(0.0)
                context_precision_scores.append(0.0)
                relevance_scores.append(raw_relevance)
            else:
                faithfulness_scores.append(raw_faith)
                context_precision_scores.append(raw_prec)
                relevance_scores.append(raw_relevance)
        # Attach per-sample adjusted scores for transparency and debugging
        dataFrame["faithfulness_adjusted"] = faithfulness_scores
        dataFrame["context_precision_adjusted"] = context_precision_scores
        dataFrame["abstained"] = [_is_abstaining(a) for a in answers]
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
        avg_context_precision = sum(context_precision_scores) / len(context_precision_scores) if context_precision_scores else 0.0
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        return avg_faithfulness, avg_context_precision, avg_relevance, dataFrame
