"""
Please refer to rdagent/log/ui/utils.py:get_summary_df for more detailed documents about metrics
"""

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit import session_state as state

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.log.submission_summary import grade_summary
from rdagent.log.ui.utils import (
    ALL,
    HIGH,
    LITE,
    MEDIUM,
    curve_figure,
    get_statistics_df,
    get_summary_df,
    lite_curve_figure,
    percent_df,
)
from rdagent.scenarios.kaggle.kaggle_crawler import get_metric_direction


def _has_mle_results_df(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    score_col = pd.to_numeric(df.get("SOTA MLE Submission Score (to_submit)"), errors="coerce")
    if score_col.notna().any():
        return True
    eval_type = df.get("Submission Eval Type")
    if eval_type is not None and eval_type.astype(str).str.strip().eq("MLE").any():
        return True
    return False


def curves_win(summary: dict):
    # draw curves
    cbwin1, cbwin2 = st.columns(2)
    if cbwin1.toggle("Show Curves", key="show_curves"):
        for k, v in summary.items():
            with st.container(border=True):
                st.markdown(f"**:blue[{k}] - :violet[{v['competition']}]**")
                try:
                    tscores = {k: v for k, v in v["test_scores"].items()}
                    tscores = pd.Series(tscores)
                    vscores = {}
                    for k, vs in v["valid_scores"].items():
                        if not isinstance(vs, pd.DataFrame) or vs.empty or vs.shape[1] == 0:
                            st.warning(f"Loop {k} has empty valid-score data. Skipping this loop in curve plot.")
                            continue
                        if not vs.index.is_unique:
                            st.warning(
                                f"Loop {k}'s valid scores index are not unique, only the last one will be kept to show."
                            )
                            st.write(vs)
                        vscores[k] = vs[~vs.index.duplicated(keep="last")].iloc[:, 0]
                    if len(vscores) == 0:
                        st.warning("No usable valid-score data found for this run. Skipping curve plot.")
                        continue
                    metric_name = list(vscores.values())[0].name
                    vscores = pd.DataFrame(vscores)
                    if "ensemble" in vscores.index:
                        ensemble_row = vscores.loc[["ensemble"]]
                        vscores = pd.concat([ensemble_row, vscores.drop("ensemble")])
                    vscores = vscores.T
                    vscores["test"] = tscores
                    vscores.index = [f"L{i}" for i in vscores.index]
                    vscores.columns.name = metric_name

                    st.plotly_chart(curve_figure(vscores))
                except Exception as e:
                    import traceback

                    st.markdown("- Error: " + str(e))
                    st.code(traceback.format_exc())
                    st.markdown("- Valid Scores: ")
                    # st.write({k: type(v) for k, v in v["valid_scores"].items()})
                    st.json(v["valid_scores"])
    if cbwin2.toggle("Show Curves (Lite)", key="show_curves_lite"):
        st.pyplot(lite_curve_figure(summary))


def get_reference_statistics_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "Has Valid Score",
        "Has Baseline",
        "Comparable",
        "Better Than Baseline",
        "Worse Than Baseline",
        "Same As Baseline",
        "Unknown Direction",
    ]
    if summary_df.empty:
        return pd.DataFrame(0.0, index=metric_cols, columns=["Kaggle Reference 统计(%)"])

    total = max(summary_df.shape[0], 1)
    valid_scores = pd.to_numeric(summary_df.get("SOTA Exp Score (valid, to_submit)"), errors="coerce")
    baseline_scores = pd.to_numeric(summary_df.get("Baseline Score"), errors="coerce")

    has_valid = valid_scores.notna()
    has_baseline = baseline_scores.notna()
    comparable = has_valid & has_baseline

    better = 0
    worse = 0
    same = 0
    unknown_direction = 0

    for idx in summary_df.index:
        if not comparable.loc[idx]:
            continue

        competition = summary_df.loc[idx, "Competition"] if "Competition" in summary_df.columns else None
        direction = summary_df.loc[idx, "Baseline Direction"] if "Baseline Direction" in summary_df.columns else None

        bigger_is_better = None
        if isinstance(direction, str):
            d = direction.strip().lower()
            if d in {"bigger_better", "higher_better", "maximize", "max"}:
                bigger_is_better = True
            elif d in {"smaller_better", "lower_better", "minimize", "min"}:
                bigger_is_better = False

        if bigger_is_better is None and isinstance(competition, str) and competition:
            try:
                bigger_is_better = bool(get_metric_direction(competition))
            except Exception:
                bigger_is_better = None

        ours = float(valid_scores.loc[idx])
        base = float(baseline_scores.loc[idx])
        if bigger_is_better is None:
            unknown_direction += 1
        elif ours == base:
            same += 1
        elif (ours > base and bigger_is_better) or (ours < base and not bigger_is_better):
            better += 1
        else:
            worse += 1

    stat_df = pd.DataFrame(index=metric_cols, columns=["Kaggle Reference 统计(%)"], dtype="float64")
    stat_df.loc["Has Valid Score", "Kaggle Reference 统计(%)"] = has_valid.sum() / total * 100
    stat_df.loc["Has Baseline", "Kaggle Reference 统计(%)"] = has_baseline.sum() / total * 100
    stat_df.loc["Comparable", "Kaggle Reference 统计(%)"] = comparable.sum() / total * 100
    stat_df.loc["Better Than Baseline", "Kaggle Reference 统计(%)"] = better / total * 100
    stat_df.loc["Worse Than Baseline", "Kaggle Reference 统计(%)"] = worse / total * 100
    stat_df.loc["Same As Baseline", "Kaggle Reference 统计(%)"] = same / total * 100
    stat_df.loc["Unknown Direction", "Kaggle Reference 统计(%)"] = unknown_direction / total * 100
    return stat_df


def all_summarize_win():
    if "summary_autogen_tried" not in state:
        state.summary_autogen_tried = set()

    def shorten_folder_name(folder: str) -> str:
        if "amlt" in folder:
            return folder[folder.rfind("amlt") + 5 :].split("/")[0]
        if "ep" in folder:
            return folder[folder.rfind("ep") :]
        return folder

    selected_folders = st.multiselect(
        "Show these folders",
        state.log_folders,
        state.log_folders,
        format_func=shorten_folder_name,
    )
    for lf in selected_folders:
        if not (Path(lf) / "summary.pkl").exists():
            if lf not in state.summary_autogen_tried:
                state.summary_autogen_tried.add(lf)
                try:
                    grade_summary(lf)
                except Exception as e:
                    st.warning(f"Auto-generate summary failed for **{lf}**: {e}")

        if not (Path(lf) / "summary.pkl").exists():
            st.warning(
                f"summary.pkl not found in **{lf}**\n\nRun: `python -m rdagent.app.cli grade_summary {lf}` (or `rdagent grade_summary {lf}`)"
            )
    summary = {}
    dfs = []
    for lf in selected_folders:
        s, df = get_summary_df(lf)
        df.index = [f"{shorten_folder_name(lf)} - {idx}" for idx in df.index]

        dfs.append(df)
        summary.update({f"{shorten_folder_name(lf)} - {k}": v for k, v in s.items()})
    if not dfs:
        st.warning("No summary data available for selected folders.")
        return

    base_df_raw = pd.concat(dfs)
    if not DS_RD_SETTING.if_using_mle_data:
        st.info("Summary page is available only when MLE mode is enabled (if_using_mle_data=true).")
        return
    if not _has_mle_results_df(base_df_raw):
        st.info("No MLE results found in selected folders. Summary page is hidden unless MLE results are available.")
        return

    def safe_mean(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns:
            return 0.0
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return 0.0
        return float(s.mean())

    valid_rate = safe_mean(base_df_raw, "Valid Improve")
    test_rate = safe_mean(base_df_raw, "Test Improve")
    submit_merge_rate = safe_mean(base_df_raw, "Submit Merge")
    merge_sota_avg = safe_mean(base_df_raw, "Merge Sota")
    base_df = percent_df(base_df_raw)
    base_df.insert(0, "Select", True)
    bt1, bt2 = st.columns(2)
    select_lite_level = bt2.selectbox(
        "Select MLE-Bench Competitions Level",
        options=["ALL", "HIGH", "MEDIUM", "LITE"],
        index=0,
        key="select_lite_level",
    )
    if select_lite_level != "ALL":
        if select_lite_level == "HIGH":
            lite_set = set(HIGH)
        elif select_lite_level == "MEDIUM":
            lite_set = set(MEDIUM)
        elif select_lite_level == "LITE":
            lite_set = set(LITE)
        else:
            lite_set = set()
        base_df["Select"] = base_df["Competition"].isin(lite_set)
    else:
        base_df["Select"] = True  # select all if ALL is chosen

    if bt1.toggle("Select Best", key="select_best"):

        def apply_func(cdf: pd.DataFrame):
            cp = base_df.loc[cdf.index[0], "Competition"]
            md = get_metric_direction(cp)
            # If SOTA Exp Score (valid, to_submit) column is empty, return the first index
            if cdf["SOTA Exp Score (valid, to_submit)"].dropna().empty:
                return cdf.index[0]
            if md:
                best_idx = cdf["SOTA Exp Score (valid, to_submit)"].idxmax()
            else:
                best_idx = cdf["SOTA Exp Score (valid, to_submit)"].idxmin()
            return best_idx

        best_idxs = base_df.groupby("Competition").apply(apply_func, include_groups=False)
        base_df["Select"] = base_df.index.isin(best_idxs.values)

    base_df = st.data_editor(
        base_df,
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", help="Stat this trace.", disabled=False),
        },
        disabled=(col for col in base_df.columns if col not in ["Select"]),
    )
    st.markdown("Ours vs Base: `math.exp(abs(math.log(sota_exp_score / baseline_score)))`")

    # 统计选择的比赛
    selected_index = base_df[base_df["Select"]].index
    base_df = base_df.loc[selected_index]
    selected_raw_df = base_df_raw.loc[selected_index]
    st.markdown(f"**统计的比赛数目: :red[{base_df.shape[0]}]**")
    if "Kaggle Score Source" in selected_raw_df.columns:
        source_counts = (
            selected_raw_df["Kaggle Score Source"].astype(str).str.strip().replace({"": "unknown"}).value_counts()
        )
        src_msg = ", ".join([f"{k}: {v}" for k, v in source_counts.items()])
        st.caption(f"Kaggle score source distribution: {src_msg}")

    missing_eval_cols = ["SOTA Exp Score (to_submit)", "SOTA Exp Score", "Best Result"]

    def has_signal_value(series: pd.Series) -> bool:
        if series is None:
            return False
        for v in series.dropna().tolist():
            if isinstance(v, str):
                if v.strip().lower() in {"", "n/a", "none", "nan"}:
                    continue
                return True
            return True
        return False

    has_eval_signal = False
    for col in missing_eval_cols:
        if col in selected_raw_df.columns and has_signal_value(selected_raw_df[col]):
            has_eval_signal = True
            break

    comparable_count = (
        pd.to_numeric(selected_raw_df.get("SOTA Exp Score (valid, to_submit)"), errors="coerce").notna()
        & pd.to_numeric(selected_raw_df.get("Baseline Score"), errors="coerce").notna()
    ).sum()

    successful_decision_count = (
        pd.to_numeric(selected_raw_df.get("Successful Final Decision", pd.Series(dtype="float64")), errors="coerce")
        .fillna(0)
        .sum()
    )
    # Enable DS reference fallback when eval artifacts are absent, but valid-vs-baseline comparison is possible.
    use_reference_fallback = not has_eval_signal and (successful_decision_count > 0 or comparable_count > 0)
    if use_reference_fallback:
        st.info(
            "No submission/evaluation artifacts were found for the selected traces (for example, missing running.submission_score logs). "
            "Switching to Kaggle-reference statistics for this data-science view."
        )
        if not DS_RD_SETTING.if_using_mle_data and "Competition" in selected_raw_df.columns:
            missing_eval_competitions = []
            for competition in selected_raw_df["Competition"].dropna().astype(str).unique().tolist():
                eval_probe = (
                    Path(DS_RD_SETTING.local_data_path) / DS_RD_SETTING.eval_sub_dir / competition / "submission_test.csv"
                )
                if not eval_probe.exists():
                    missing_eval_competitions.append(f"{competition} -> missing {eval_probe}")

            if missing_eval_competitions:
                st.warning("Test-eval is disabled for some competitions in the current data-science setup:")
                for item in missing_eval_competitions:
                    st.write(f"- {item}")

    stat_win_left, stat_win_right = st.columns(2)
    with stat_win_left:
        if use_reference_fallback:
            ref_stat_df = get_reference_statistics_df(selected_raw_df)
            st.dataframe(ref_stat_df.round(2))
            markdown_table = f"""
    | Kaggle Ref | {ref_stat_df.iloc[0,0]:.1f} | {ref_stat_df.iloc[1,0]:.1f} | {ref_stat_df.iloc[2,0]:.1f} | {ref_stat_df.iloc[3,0]:.1f} | {ref_stat_df.iloc[4,0]:.1f} | {ref_stat_df.iloc[5,0]:.1f} | {ref_stat_df.iloc[6,0]:.1f} |
    | Valid Improve {valid_rate * 100:.2f}% | Test Improve {test_rate * 100:.2f}% | Submit Merge {submit_merge_rate * 100:.2f}% | Merge Sota {merge_sota_avg * 100:.2f}% |
    """
        else:
            stat_df = get_statistics_df(base_df)
            st.dataframe(stat_df.round(2))
            markdown_table = f"""
| SOTA Exp | {stat_df.iloc[0,1]:.1f} | {stat_df.iloc[1,1]:.1f} | {stat_df.iloc[2,1]:.1f} | {stat_df.iloc[3,1]:.1f} | {stat_df.iloc[4,1]:.1f} | {stat_df.iloc[5,1]:.1f} | {stat_df.iloc[6,1]:.1f} |
| Valid Improve {valid_rate * 100:.2f}% | Test Improve {test_rate * 100:.2f}% | Submit Merge {submit_merge_rate * 100:.2f}% | Merge Sota {merge_sota_avg * 100:.2f}% |
"""
        st.text(markdown_table)
    with stat_win_right:
        loop_counts = pd.to_numeric(selected_raw_df.get("Total Loops", pd.Series(dtype="float64")), errors="coerce").dropna()

        if loop_counts.empty:
            st.info("No valid loop-count data to visualize.")
        elif loop_counts.nunique() <= 1:
            st.info(
                "Total Loops has only one unique value in the selected data, so distribution is degenerate and is not plotted."
            )
            st.write(f"Unique value: {loop_counts.iloc[0]} (count={len(loop_counts)})")
        else:
            # Create histogram
            fig = px.histogram(
                loop_counts,
                nbins=15,
                title="Distribution of Total Loops",
                color_discrete_sequence=["#3498db"],
            )
            fig.update_layout(title_font_size=16, title_font_color="#2c3e50")

            # Calculate statistics
            mean_value = float(loop_counts.mean())
            median_value = float(loop_counts.median())

            # Add mean and median lines
            fig.add_vline(x=mean_value, line_color="#e74c3c", line_width=3)
            fig.add_vline(x=median_value, line_color="#f39c12", line_width=3)

            fig.add_annotation(
                x=0.02,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"<span style='color:#e74c3c; font-weight:bold'>Mean: {mean_value:.1f}</span><br><span style='color:#f39c12; font-weight:bold'>Median: {median_value:.1f}</span>",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(128,128,128,0.5)",
                borderwidth=1,
                font=dict(size=12, color="#333333"),
            )

            st.plotly_chart(fig, width="stretch")

    # write curve
    st.subheader("Curves", divider="rainbow")
    curves_win(summary)


with st.container(border=True):
    try:
        all_summarize_win()
    except Exception as e:
        import traceback

        st.error(f"Error occurred when show summary:\n{e}")
        st.code(traceback.format_exc())
