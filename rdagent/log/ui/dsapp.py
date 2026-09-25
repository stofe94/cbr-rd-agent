import argparse
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit import session_state as state

from rdagent.app.data_science.loop import DataScienceRDLoop
from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.log.submission_summary import cleanup_debug_only_timestamped_log_runs
from rdagent.log.ui.conf import UI_SETTING
from rdagent.log.ui.utils import get_summary_df
from rdagent.log.utils import extract_json


parser = argparse.ArgumentParser(description="RD-Agent Data-Science Streamlit App")
parser.add_argument("--log_dir", type=str, help="Path to the log directory")
args = parser.parse_args()


def convert_log_folder_str(lf: str) -> str:
    if "/" not in lf:
        return f"{UI_SETTING.amlt_path}/{lf.strip()}/combined_logs"
    return lf.strip()


def extract_amlt_name(x: str) -> str:
    if "amlt" not in x:
        return x
    return x[x.rfind("amlt") + 5 :].split("/")[0]


def _has_live_kaggle_data(log_folders: list[str]) -> bool:
    for folder in log_folders:
        root = Path(folder)
        if not root.exists():
            continue
        for p in root.glob("**/running/submission_score/*/*.pkl"):
            try:
                obj = pickle.loads(p.read_bytes())
                report = extract_json(obj)
            except Exception:
                continue
            if not isinstance(report, dict):
                continue
            if str(report.get("reference_type", "")).strip().lower() == "kaggle_live_api":
                return True
            if str(report.get("kaggle_status", "")).strip() != "":
                return True
    return False


def _has_aide_files() -> bool:
    aide_root = Path(UI_SETTING.aide_path)
    if not aide_root.exists():
        return False
    return any(aide_root.rglob("**/filtered_journal.json"))


def _has_mle_results(log_folders: list[str]) -> bool:
    for folder in log_folders:
        summary_path = Path(folder) / "summary.pkl"
        if not summary_path.exists():
            continue

        try:
            _, df = get_summary_df(folder)
        except Exception:
            continue

        if df.empty:
            continue

        score_col = pd.to_numeric(df.get("SOTA MLE Submission Score (to_submit)"), errors="coerce")
        if score_col.notna().any():
            return True

        eval_type = df.get("Submission Eval Type")
        if eval_type is not None and eval_type.astype(str).str.strip().eq("MLE").any():
            return True

    return False


# 设置主日志路径
if "log_folder" not in state:
    if args.log_dir:
        state.log_folder = Path(convert_log_folder_str(args.log_dir))
    else:
        state.log_folder = Path("./log")
if "log_folders" not in state:
    if args.log_dir:
        state.log_folders = [convert_log_folder_str(args.log_dir)]
    else:
        state.log_folders = [convert_log_folder_str(i) for i in UI_SETTING.default_log_folders]

for lf in state.log_folders:
    try:
        cleanup_debug_only_timestamped_log_runs(lf)
    except Exception:
        # Keep UI robust even if cleanup fails for a folder.
        pass

trace_page = st.Page("ds_trace.py", title="Trace", icon="📈")
summary_page = st.Page("ds_summary.py", title="Summary", icon="📊")
live_kaggle_page = st.Page("ds_live_kaggle.py", title="Live Kaggle", icon="🛰️")
aide_page = st.Page("aide.py", title="Aide", icon="🧑‍🏫")

show_summary = bool(DS_RD_SETTING.if_using_mle_data) and _has_mle_results(state.log_folders)
show_live_kaggle = _has_live_kaggle_data(state.log_folders)
show_aide = _has_aide_files()

pages = [trace_page]
if show_summary:
    pages.insert(0, summary_page)
if show_live_kaggle:
    pages.insert(1 if show_summary else 0, live_kaggle_page)
if show_aide:
    pages.append(aide_page)

st.set_page_config(layout="wide", page_title="RD-Agent", page_icon="🎓", initial_sidebar_state="expanded")
st.navigation(pages).run()


# UI - Sidebar
with st.sidebar:
    st.subheader("Pages", divider="rainbow")
    st.page_link(trace_page, icon="📈")
    if show_summary:
        st.page_link(summary_page, icon="📊")
    if show_live_kaggle:
        st.page_link(live_kaggle_page, icon="🛰️")
    if show_aide:
        st.page_link(aide_page, icon="🧑‍🏫")

    st.subheader("Settings", divider="rainbow")
    with st.form("log_folder_form", border=False):
        log_folder_str = st.text_area(
            "**Log Folders**(split by ';')", value=";".join(extract_amlt_name(i) for i in state.log_folders)
        )
        if st.form_submit_button("Confirm"):
            state.log_folders = [
                convert_log_folder_str(folder) for folder in log_folder_str.split(";") if folder.strip()
            ]
            st.rerun()
