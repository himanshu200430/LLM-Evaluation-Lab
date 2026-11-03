"""Print README-ready markdown tables for saved experiments.

Usage:
    python make_report.py 8 9 10
        (experiment ids; see them with: python history.py)

    python make_report.py
        (the newest experiment)
"""

import sys

from app.db.crud import (
    get_faithfulness,
    get_judged_results,
    list_experiments,
)
from app.services.report import experiment_markdown


experiments = {
    experiment["id"]: experiment
    for experiment in list_experiments()
}

if not experiments:
    sys.exit("No experiments saved yet.")


ids = [int(arg) for arg in sys.argv[1:]] or [max(experiments)]

for experiment_id in ids:
    if experiment_id not in experiments:
        print(f"(no experiment #{experiment_id})\n")
        continue

    print(
        experiment_markdown(
            experiments[experiment_id],
            get_judged_results(experiment_id),
            get_faithfulness(experiment_id),
        )
    )

    print()