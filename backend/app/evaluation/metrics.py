from collections import defaultdict
def summarize(results):
    total = len(results)

    if total == 0:
        return {
            "total": 0,
            "accuracy_pct": 0.0,
            "strict_accuracy_pct": 0.0,
            "errors": 0,
            "avg_latency_sec": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
            "total_cost_usd": 0.0
        }
    correct = sum(1 for r in results if r["correct"])
    strict = sum(1 for r in results if r["normalized"])

    return {
        "total": total,
        "accuracy_pct": round(correct / total * 100, 1),
        "strict_accuracy_pct": round(strict / total * 100, 1),
        "errors": sum(1 for r in results if r["error"]),
        "avg_latency_sec": round(
            sum(r["latency_sec"] for r in results) / total, 2
        ),
        "tokens_in": sum(r["tokens_in"] for r in results),
        "tokens_out": sum(r["tokens_out"] for r in results),
        "total_cost_usd": round(
            sum(r["cost_usd"] for r in results), 6
        )
    }

def summarize_by_model(results):
    groups = defaultdict(list)
    for r in results:
        groups[r["model"]].append(r)
    return {model: summarize(rows) for model, rows in groups.items()}
