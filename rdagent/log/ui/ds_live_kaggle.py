from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit import session_state as state

from rdagent.log.submission_summary import grade_summary
from rdagent.log.ui.utils import get_summary_df


def _shorten_folder_name(folder: str) -> str:
	if "amlt" in folder:
		return folder[folder.rfind("amlt") + 5 :].split("/")[0]
	if "ep" in folder:
		return folder[folder.rfind("ep") :]
	return folder


def _load_combined_summary(selected_folders: list[str]) -> pd.DataFrame:
	dfs: list[pd.DataFrame] = []
	for lf in selected_folders:
		summary_path = Path(lf) / "summary.pkl"
		if not summary_path.exists():
			if "live_summary_autogen_tried" not in state:
				state.live_summary_autogen_tried = set()
			if lf not in state.live_summary_autogen_tried:
				state.live_summary_autogen_tried.add(lf)
				try:
					grade_summary(lf)
				except Exception as e:
					st.warning(f"Auto-generate summary failed for {lf}: {e}")

		_, df = get_summary_df(lf)
		if df.empty:
			continue
		df = df.copy()
		df.index = [f"{_shorten_folder_name(lf)} - {idx}" for idx in df.index]
		dfs.append(df)

	if not dfs:
		return pd.DataFrame()
	return pd.concat(dfs)


def _render_kpis(df: pd.DataFrame) -> None:
	total = len(df)
	with_score = pd.to_numeric(df.get("SOTA LiveKaggle Submission Score (to_submit)"), errors="coerce").notna().sum()
	if with_score == 0:
		with_score = pd.to_numeric(df.get("SOTA Exp Score (to_submit)"), errors="coerce").notna().sum()
	pending_or_missing = (df.get("Kaggle Status", pd.Series(index=df.index, dtype="object")).fillna("").str.len() == 0).sum()
	completed = df.get("Kaggle Status", pd.Series(index=df.index, dtype="object")).fillna("").str.contains(
		"complete", case=False
	).sum()

	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Runs", total)
	c2.metric("LiveKaggle Scored Runs", int(with_score))
	c3.metric("Complete Status", int(completed))
	c4.metric("Missing Status", int(pending_or_missing))


def _render_charts(df: pd.DataFrame) -> None:
	left, right = st.columns(2)

	with left:
		status_series = df.get("Kaggle Status", pd.Series(index=df.index, dtype="object")).fillna("unknown")
		status_df = status_series.value_counts().rename_axis("status").reset_index(name="count")
		fig_status = px.bar(
			status_df,
			x="status",
			y="count",
			color="status",
			title="Submission Status Distribution",
		)
		fig_status.update_layout(showlegend=False)
		st.plotly_chart(fig_status, width="stretch")

	with right:
		source_series = df.get("Kaggle Score Source", pd.Series(index=df.index, dtype="object")).fillna("unknown")
		source_df = source_series.value_counts().rename_axis("source").reset_index(name="count")
		fig_source = px.pie(
			source_df,
			names="source",
			values="count",
			hole=0.45,
			title="Score Source Mix",
		)
		st.plotly_chart(fig_source, width="stretch")

	score_col = "SOTA LiveKaggle Submission Score (to_submit)"
	if score_col not in df.columns:
		score_col = "SOTA Exp Score (to_submit)"
	comp_df = df[["Competition", score_col, "Kaggle Best Score", "Kaggle Quality Class"]].copy()
	comp_df[score_col] = pd.to_numeric(comp_df[score_col], errors="coerce")
	comp_df["Kaggle Best Score"] = pd.to_numeric(comp_df["Kaggle Best Score"], errors="coerce")
	comp_df = comp_df.dropna(subset=[score_col, "Kaggle Best Score"])

	if comp_df.empty:
		st.info("No scored live-eval rows available for score comparison charts yet.")
		return

	cc1, cc2 = st.columns(2)
	with cc1:
		fig_scatter = px.scatter(
			comp_df,
			x="Kaggle Best Score",
			y=score_col,
			color="Kaggle Quality Class",
			hover_data=["Competition"],
			title="LiveKaggle Submission Score vs Kaggle Best",
		)
		fig_scatter.update_layout(xaxis_title="Kaggle Best Score", yaxis_title="LiveKaggle Submission Score")
		st.plotly_chart(fig_scatter, width="stretch")

	with cc2:
		gap_df = df[["Competition", "Ours - Best", "Kaggle Quality Class"]].copy()
		gap_df["Ours - Best"] = pd.to_numeric(gap_df["Ours - Best"], errors="coerce")
		gap_df = gap_df.dropna(subset=["Ours - Best"]).sort_values("Ours - Best", ascending=False)
		if gap_df.empty:
			st.info("No Ours-Best gap values available yet.")
		else:
			fig_gap = px.bar(
				gap_df,
				x="Competition",
				y="Ours - Best",
				color="Kaggle Quality Class",
				title="Score Gap to Kaggle Best",
			)
			fig_gap.update_layout(xaxis_tickangle=-30)
			st.plotly_chart(fig_gap, width="stretch")


def _render_table(df: pd.DataFrame) -> None:
	show_cols = [
		"Competition",
		"SOTA LID (to_submit)",
		"SOTA LiveKaggle Submission Score (to_submit)",
		"SOTA MLE Submission Score (to_submit)",
		"Submission Eval Type",
		"Kaggle Status",
		"Kaggle Ref",
		"Kaggle Created At",
		"Kaggle Score Source",
		"Kaggle Reference Type",
		"Kaggle Quality Class",
		"Kaggle Best Score",
		"Ours - Best",
		"Ours vs Best",
		"Bronze Threshold",
		"Silver Threshold",
		"Gold Threshold",
		"Medium Threshold",
	]
	present_cols = [c for c in show_cols if c in df.columns]
	table_df = df[present_cols].copy()
	if "Kaggle Created At" in table_df.columns:
		table_df["Kaggle Created At"] = pd.to_datetime(table_df["Kaggle Created At"], errors="coerce")
		table_df = table_df.sort_values("Kaggle Created At", ascending=False, na_position="last")

	st.dataframe(table_df, width="stretch")


st.title("Live Kaggle Eval Summary")
st.caption("Dedicated dashboard for live Kaggle submission outcomes. Live scores are explicitly labeled as LiveKaggle Submission Score.")

default_folders = state.log_folders if "log_folders" in state else ["./log"]
selected_folders = st.multiselect(
	"Show these folders",
	default_folders,
	default_folders,
	format_func=_shorten_folder_name,
)

combined_df = _load_combined_summary(selected_folders)
if combined_df.empty:
	st.warning("No summary data available for selected folders.")
else:
	_render_kpis(combined_df)
	st.divider()
	_render_charts(combined_df)
	st.divider()
	st.subheader("Live Eval Detail Table")
	_render_table(combined_df)
