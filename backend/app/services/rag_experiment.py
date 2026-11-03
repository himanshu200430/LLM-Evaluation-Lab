"""RAG experiments for the dashboard: a retrieval grid, and the same model with and without RAG."""

from app.db.crud import (
    get_results,
    save_experiment,
    save_faithfulness,
)
from app.evaluation.faithfulness import judge_faithfulness
from app.rag.pipeline import RagProvider, build_index
from app.rag.retrieval_metrics import hit_rate
from app.services.experiment_runner import run_experiment


def hit_rate_grid(doc_path, items, chunk_sizes, ks, embeddings=None):
    """Retrieval hit rate for every (chunk size, k) pair. Makes no LLM calls."""

    rows = []

    for chunk_size in chunk_sizes:
        store = build_index(
            doc_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 8,
            embeddings=embeddings,
        )

        for k in ks:
            result = hit_rate(store, items, k)

            rows.append(
                {
                    "chunk_size": chunk_size,
                    "k": k,
                    "chunks": store.count(),
                    "hit_rate_pct": result["hit_rate_pct"],
                    "hits": result["hits"],
                    "total": result["total"],
                    "misses": result["misses"],
                }
            )

    return rows


def run_rag_experiment(
    items,
    doc_path,
    llm,
    chunk_size,
    k,
    dataset_name,
    embeddings=None,
    judge=None,
    delay_sec=0.5,
    on_progress=None,
):
    """
    Ask the same questions to `llm` without RAG and with RAG,
    and save both as one experiment.

    If `judge` is given, also check whether each RAG answer
    is supported by its retrieved context.
    """

    store = build_index(
        doc_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 8,
        embeddings=embeddings,
    )

    rag = RagProvider(llm, store, k=k)

    results = run_experiment(
        items,
        [llm, rag],
        delay_sec=delay_sec,
    )

    experiment_id = save_experiment(
        dataset_name,
        f"rag(chunk={chunk_size},k={k})",
        results,
    )

    faithful_checked = 0
    faithful_failed = 0

    if judge is not None:
        saved_ids = {
            (row["model"], row["question"]): row["id"]
            for row in get_results(experiment_id)
        }

        rag_rows = [
            row
            for row in results
            if row["model"] == rag.name and not row["error"]
        ]

        faith_rows = []

        for i, row in enumerate(rag_rows, start=1):
            context = rag.contexts.get(row["question"], "")

            verdict, error = judge_faithfulness(
                judge,
                row["question"],
                context,
                row["answer"],
            )

            faith_rows.append(
                {
                    "result_id": saved_ids[
                        (row["model"], row["question"])
                    ],
                    "context": context,
                    "faithful": verdict.faithful if verdict else None,
                    "reason": verdict.reason if verdict else None,
                    "error": error,
                }
            )

            faithful_failed += verdict is None

            if on_progress:
                on_progress(i, len(rag_rows))

        save_faithfulness(judge.name, faith_rows)

        faithful_checked = len(rag_rows) - faithful_failed

    return {
        "experiment_id": experiment_id,
        "rag_name": rag.name,
        "faithful_checked": faithful_checked,
        "faithful_failed": faithful_failed,
    }