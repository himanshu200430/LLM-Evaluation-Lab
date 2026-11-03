from app.evaluation.exact_match import contains_match

def hit_rate(store,  items, k=3):
    retriever = store.retriever(k)
    misses = []

    for item in items:
        documents = retriever.invoke(item.question)

        if not any(
            contains_match(document.page_content, item.expected_answer)
            for document in documents
        ):
            misses.append(item.question)
    
    total = len(items)
    hits = total - len(misses)

    return {
        "hits": hits,
        "total": total,
        "hit_rate_pct": round(hits / total * 100, 1) if total else 0.0,
        "misses": misses,
    }

