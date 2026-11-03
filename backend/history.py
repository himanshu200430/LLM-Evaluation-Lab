import sys

from app.db.crud import get_results, list_experiments
from app.evaluation.metrics import summarize_by_model

if len(sys.argv) == 1:
    for e in list_experiments():
        print(f"#{e['id']} {e['created_at']} {e['dataset_name']} {e['mode']} {e['total_questions']} questions")
else:
    results = get_results(int(sys.argv[1]))
    for model, s in summarize_by_model(results).items():
        print(f"{model:<34} acc={s['accuracy_pct']}% strict={s['strict_accuracy_pct']}% "
        f"latency={s['avg_latency_sec']}s cost=${s['total_cost_usd']} errors={s['errors']}")