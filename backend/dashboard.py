"""LLM EvalLab dashboard.   Run with:  streamlit run dashboard.py"""
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from app.db.crud import get_faithfulness, get_judged_results, list_experiments, save_experiment
from app.providers.registry import (PRESETS, all_model_names, build_provider, build_providers,
                                    delete_custom_spec, load_custom_specs, save_custom_spec)
from app.rag.embeddings import ChromaDefaultEmbeddings
from app.services.dataset_parser import DatasetError, parse_dataset
from app.services.experiment_runner import run_experiment
from app.services.judging import judge_experiment
from app.services.rag_experiment import hit_rate_grid, run_rag_experiment
from app.services.report import answers_table, faithfulness_summary, model_summary

DATA = Path(__file__).resolve().parents[1] / "data"

st.set_page_config(page_title="LLM EvalLab", layout="wide")


def local_datasets() -> list[str]:
    """Question files in the data folder (validation files are not question sets)."""
    files = sorted(list(DATA.glob("*.csv")) + list(DATA.glob("*.jsonl")))
    return [f.name for f in files if "validation" not in f.name]


def list_documents() -> list[str]:
    """Documents in data/documents that RAG can read."""
    folder = DATA / "documents"
    files = [f for f in folder.glob("*") if f.suffix.lower() in (".txt", ".md", ".pdf")] if folder.exists() else []
    return sorted(f.name for f in files)


@st.cache_resource(show_spinner="Loading the embedding model (the very first run downloads about 80 MB)...")
def get_embeddings():
    return ChromaDefaultEmbeddings()


def summary_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(model_summary(rows)).set_index("model")
    frame["cost_per_1k_questions_usd"] = (frame["total_cost_usd"] / frame["total"] * 1000).round(4)
    columns = ["accuracy_pct", "strict_accuracy_pct", "judge_accuracy_pct", "avg_judge_score",
               "avg_latency_sec", "total_cost_usd", "cost_per_1k_questions_usd", "errors", "total", "judged"]
    return frame[columns]


# ------------------------------------------------------------------ page 1
def page_run():
    st.header("Run an experiment")
    st.caption("Send the same questions to one or more models, then compare them on the Results page.")

    left, right = st.columns(2)
    with left:
        uploaded = st.file_uploader("Upload a dataset (.csv or .jsonl)", type=["csv", "jsonl"])
    with right:
        names = local_datasets()
        local = st.selectbox("...or pick one from the data folder", names) if names else None

    if uploaded is not None:
        name, content = uploaded.name, uploaded.getvalue()
    elif local:
        name, content = local, (DATA / local).read_bytes()
    else:
        st.info("Upload a dataset to begin.")
        return

    try:
        items = parse_dataset(name, content)
    except DatasetError as e:
        st.error(f"Dataset problem: {e}")
        return

    st.success(f"{len(items)} questions loaded from {name}")
    with st.expander("Preview"):
        st.dataframe(pd.DataFrame([i.model_dump() for i in items]).head(10), use_container_width=True)

    models = st.multiselect("Models", all_model_names(), default=["Groq"],
                            help="Add more models on the Models page.")
    max_questions = st.number_input("Max questions to use", min_value=1, max_value=len(items),
                                    value=min(len(items), 20))
    mode = st.radio("Mode", ["One call per question", "Batch of 10 questions per call"],
                    help="Batch uses far fewer API calls, but latency and tokens become estimates.")
    batch = mode.startswith("Batch")

    calls = len(models) * (-(-int(max_questions) // 10) if batch else int(max_questions))
    st.caption(f"This will make about {calls} API calls.")

    if st.button("Run experiment", type="primary", disabled=not models):
        try:
            providers = build_providers(models)
        except ValueError as e:
            st.error(str(e))
            return
        with st.spinner("Asking the models..."):
            results = run_experiment(items[: int(max_questions)], providers, batch_size=10 if batch else None)
        experiment_id = save_experiment(name, "batch(10)" if batch else "per-question", results)
        st.session_state["last_experiment"] = experiment_id
        st.success(f"Saved as experiment #{experiment_id}. Open the Results page for details.")
        st.dataframe(summary_frame(get_judged_results(experiment_id)), use_container_width=True)


# ------------------------------------------------------------------ page 2
def page_results():
    st.header("Results and history")
    experiments = list_experiments()
    if not experiments:
        st.info("No experiments yet. Run one on the first page.")
        return

    labels = {e["id"]: f"#{e['id']}  {e['created_at']}  {e['dataset_name']}  [{e['mode']}]" for e in experiments}
    ids = list(labels)
    last = st.session_state.get("last_experiment")
    experiment_id = st.selectbox("Experiment", ids, index=ids.index(last) if last in ids else 0,
                                 format_func=lambda i: labels[i])

    rows = get_judged_results(experiment_id)
    if not rows:
        st.warning("This experiment has no results.")
        return

    st.subheader("Model comparison")
    frame = summary_frame(rows)
    st.dataframe(frame, use_container_width=True)

    accuracy_cols = ["accuracy_pct"] + (["judge_accuracy_pct"] if frame["judge_accuracy_pct"].notna().any() else [])
    c1, c2, c3 = st.columns(3)
    c1.caption("Accuracy %")
    c1.bar_chart(frame[accuracy_cols])
    c2.caption("Average latency (seconds)")
    c2.bar_chart(frame[["avg_latency_sec"]])
    c3.caption("Total estimated cost (USD)")
    c3.bar_chart(frame[["total_cost_usd"]])

    # ---- LLM judge ----
    with st.expander("LLM judge (a second model grades each answer)"):
        n = sum(1 for r in rows if not r["error"])
        st.write(f"Judging uses about {n} API calls. Run it again any time; the newest verdict is kept.")
        judge_name = st.selectbox("Judge model", all_model_names(),
                                  index=all_model_names().index("Groq"),
                                  help="Pick a different model from the one being judged, to reduce self-preference bias.")
        if st.button("Judge this experiment"):
            try:
                judge = build_provider(judge_name)
            except ValueError as e:
                st.error(str(e))
            else:
                bar = st.progress(0.0)
                ok, failed = judge_experiment(experiment_id, judge,
                                              on_progress=lambda done, total: bar.progress(done / total))
                st.success(f"Judged {ok} answers ({failed} failed).")
                st.rerun()

    # ---- answers ----
    st.subheader("Answers")
    table = pd.DataFrame(answers_table(rows))
    f1, f2, f3 = st.columns(3)
    chosen = f1.multiselect("Models", sorted(table["model"].unique()), default=sorted(table["model"].unique()))
    only_wrong = f2.checkbox("Only wrong answers (string match or judge)")
    only_disagree = f3.checkbox("Only where string match and judge disagree")

    view = table[table["model"].isin(chosen)]
    if only_wrong:
        view = view[(~view["string_ok"]) | (view["judge_ok"] == False)]  # noqa: E712
    if only_disagree:
        view = view[view["judge_ok"].notna() & (view["string_ok"] != view["judge_ok"].astype("boolean"))]
    st.dataframe(view, use_container_width=True)
    st.download_button("Download all answers (CSV)", table.to_csv(index=False), file_name=f"experiment_{experiment_id}.csv")

    # ---- faithfulness (only for RAG experiments that were checked) ----
    faith = get_faithfulness(experiment_id)
    if faith:
        st.subheader("Faithfulness (is the answer supported by the retrieved context?)")
        st.dataframe(pd.DataFrame(faithfulness_summary(faith)), use_container_width=True)
        bad = [r for r in faith if r["faithful"] == 0]
        st.write(f"{len(bad)} answers were not supported by their context:")
        for r in bad:
            with st.expander(r["question"]):
                st.write(f"**Answer:** {r['answer']}")
                st.write(f"**Judge:** {r['reason']}")
                st.text(r["context"])


# ------------------------------------------------------------------ page 3
def page_models():
    st.header("Models")
    st.write("Gemini and Groq are built in (their keys and model ids come from your .env file). "
             "You can add any other model that has an OpenAI-compatible API: OpenRouter, OpenAI, DeepSeek, "
             "Together, a local Ollama server, and more.")

    specs = load_custom_specs()
    if specs:
        st.subheader("Your custom models")
        for spec in specs:
            c1, c2 = st.columns([5, 1])
            c1.write(f"**{spec['name']}**  ·  model `{spec['model']}`  ·  {spec['base_url']}  ·  "
                     f"${spec['price_in']} in / ${spec['price_out']} out per 1M tokens")
            if c2.button("Delete", key=f"delete_{spec['name']}"):
                delete_custom_spec(spec["name"])
                st.rerun()

    st.subheader("Test a model")
    test_name = st.selectbox("Model", all_model_names(), key="test_model")
    if st.button("Send a test question"):
        try:
            provider = build_provider(test_name)
        except ValueError as e:
            st.error(str(e))
        else:
            reply = provider.ask("What is the capital of France?")
            if reply.error:
                st.error(reply.error[:400])
            else:
                st.success(f"{reply.answer}  ({reply.latency_sec}s)")

    st.subheader("Add a model")
    preset = st.selectbox("Provider preset", list(PRESETS) + ["Other"])
    base_default, key_default = PRESETS.get(preset, ("", ""))
    name = st.text_input("Display name (shown in tables)", key=f"name_{preset}")
    model = st.text_input("Model id, exactly as the provider names it", key=f"model_{preset}")
    base_url = st.text_input("Base URL", value=base_default, key=f"url_{preset}")
    key_env = st.text_input("Name of the .env variable that holds the API key (empty for local servers)",
                            value=key_default, key=f"keyenv_{preset}")
    c1, c2 = st.columns(2)
    price_in = c1.number_input("Input price (USD per 1M tokens)", min_value=0.0, value=0.0, step=0.05, key=f"pin_{preset}")
    price_out = c2.number_input("Output price (USD per 1M tokens)", min_value=0.0, value=0.0, step=0.05, key=f"pout_{preset}")
    st.caption("Take the prices from the provider's pricing page. With 0 the estimated cost stays 0.")

    if st.button("Save model", type="primary"):
        try:
            saved = save_custom_spec({"name": name, "model": model, "base_url": base_url,
                                      "api_key_env": key_env, "price_in": price_in, "price_out": price_out})
        except ValueError as e:
            st.error(str(e))
        else:
            st.toast(f"Saved {saved['name']}. Add its key to .env if you have not yet.")
            st.rerun()


# ------------------------------------------------------------------ page 4
def page_rag():
    st.header("RAG lab")
    st.write("RAG = the model gets the most relevant chunks of your document together with the question. "
             "Step 1 checks how well different settings find the right chunk (no API calls). "
             "Step 2 compares the same model with and without RAG.")

    # ---- inputs: a document and a question set ----
    left, right = st.columns(2)
    with left:
        doc_upload = st.file_uploader("Upload a document (.txt, .md or .pdf)", type=["txt", "md", "pdf"])
        docs = list_documents()
        doc_choice = st.selectbox("...or pick one from data/documents", docs, key="rag_doc") if docs else None
    with right:
        data_upload = st.file_uploader("Upload questions (.csv or .jsonl)", type=["csv", "jsonl"], key="rag_data_upload")
        names = local_datasets()
        default = names.index("rag_questions.csv") if "rag_questions.csv" in names else 0
        data_choice = st.selectbox("...or pick a question set", names, index=default, key="rag_data") if names else None

    if doc_upload is not None:
        folder = Path(tempfile.gettempdir()) / "evallab_uploads"
        folder.mkdir(exist_ok=True)
        doc_path = folder / doc_upload.name
        doc_path.write_bytes(doc_upload.getvalue())
    elif doc_choice:
        doc_path = DATA / "documents" / doc_choice
    else:
        st.info("Upload a document to begin.")
        return

    if data_upload is not None:
        data_name, data_bytes = data_upload.name, data_upload.getvalue()
    elif data_choice:
        data_name, data_bytes = data_choice, (DATA / data_choice).read_bytes()
    else:
        st.info("Upload a question set to begin.")
        return
    try:
        items = parse_dataset(data_name, data_bytes)
    except DatasetError as e:
        st.error(f"Question set problem: {e}")
        return
    st.success(f"Document: {doc_path.name}   |   {len(items)} questions from {data_name}")

    # ---- step 1: retrieval grid ----
    st.subheader("Step 1: which settings find the right chunk? (free, no API calls)")
    st.caption("Hit rate = the share of questions whose expected answer text appears in the retrieved chunks. "
               "It is a simple proxy, best for short factual answers.")
    c1, c2 = st.columns(2)
    sizes = c1.multiselect("Chunk sizes (characters)", [100, 150, 200, 300, 400, 600, 800], default=[150, 300, 600])
    ks = c2.multiselect("Top-k (chunks sent to the model)", [1, 2, 3, 5, 8], default=[1, 3, 5])
    if st.button("Compute hit rate", disabled=not (sizes and ks)):
        with st.spinner("Splitting, embedding and searching..."):
            rows = hit_rate_grid(doc_path, items, sorted(sizes), sorted(ks), embeddings=get_embeddings())
        st.session_state["rag_grid"] = rows

    grid = st.session_state.get("rag_grid")
    if grid:
        frame = pd.DataFrame(grid)
        pivot = frame.pivot(index="chunk_size", columns="k", values="hit_rate_pct")
        pivot.columns = [f"k={k}" for k in pivot.columns]
        g1, g2 = st.columns(2)
        g1.dataframe(pivot, use_container_width=True)
        g2.bar_chart(pivot)
        options = [f"chunk size {r['chunk_size']}, k={r['k']}" for r in grid]
        pick = st.selectbox("Which questions did a setting miss?", options)
        missed = grid[options.index(pick)]["misses"]
        st.write("No misses." if not missed else "Missed: " + " | ".join(missed))

    # ---- step 2: RAG vs no RAG ----
    st.subheader("Step 2: the same model with and without RAG")
    a, b, c = st.columns(3)
    chunk_size = a.selectbox("Chunk size", [100, 150, 200, 300, 400, 600, 800], index=3)
    k = int(b.number_input("Top-k", min_value=1, max_value=10, value=3))
    models = all_model_names()
    llm_name = c.selectbox("Model", models, index=models.index("Groq"), key="rag_llm")
    max_questions = int(st.number_input("Max questions", min_value=1, max_value=len(items),
                                        value=min(len(items), 20), key="rag_max"))
    check = st.checkbox("Also check faithfulness (is each RAG answer supported by its chunks?)")
    judge_name = st.selectbox("Judge model", models, index=models.index("Groq"), key="rag_judge", disabled=not check)
    st.caption(f"This will make about {2 * max_questions + (max_questions if check else 0)} API calls.")

    if st.button("Run RAG vs no RAG", type="primary"):
        try:
            llm = build_provider(llm_name)
            judge = build_provider(judge_name) if check else None
        except ValueError as e:
            st.error(str(e))
            return
        bar = st.progress(0.0, text="Asking the model with and without RAG...")
        out = run_rag_experiment(items[:max_questions], doc_path, llm, chunk_size, k, data_name,
                                 embeddings=get_embeddings(), judge=judge,
                                 on_progress=lambda done, total: bar.progress(done / total, text="Checking faithfulness..."))
        bar.empty()
        st.session_state["last_experiment"] = out["experiment_id"]
        st.success(f"Saved as experiment #{out['experiment_id']}. Open the Results page for the full tables.")
        st.dataframe(summary_frame(get_judged_results(out["experiment_id"])), use_container_width=True)
        if check:
            st.dataframe(pd.DataFrame(faithfulness_summary(get_faithfulness(out["experiment_id"]))),
                         use_container_width=True)
            if out["faithful_failed"]:
                st.warning(f"{out['faithful_failed']} faithfulness checks failed (judge error).")


# ------------------------------------------------------------------ app
page = st.sidebar.radio("Page", ["Run experiment", "Results", "RAG lab", "Models"])
st.sidebar.caption("LLM EvalLab: compare models, judge answers, test RAG.")
{"Run experiment": page_run, "Results": page_results, "RAG lab": page_rag, "Models": page_models}[page]()