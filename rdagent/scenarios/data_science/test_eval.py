import json
import math
import time
import uuid
from abc import abstractmethod
from pathlib import Path

import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.conf import get_ds_env
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.kaggle.kaggle_crawler import get_metric_direction


SUBMISSION_SCORE_FILE = "submission_score.txt"
SUBMISSION_FORMAT_CHECK_OUTPUT = "test/submission_format_check.output"


class NoTestEvalError(Exception):
    """Test evaluation is not provided"""


class TestEvalBase:
    """Evaluate a workspace on Test Dataset"""

    @abstractmethod
    def eval(self, competition: str, workspace: FBWorkspace) -> str:
        """eval the workspace as competition, and return the final evaluation result"""

    @abstractmethod
    def valid(self, competition: str, workspace: FBWorkspace) -> tuple[str, int]:
        """eval the workspace as competition, and return the final format check result"""

    @abstractmethod
    def enabled(self, competition) -> bool:
        """support `eval` & `valid` or not"""

    @abstractmethod
    def get_sample_submission_name(self, competition: str) -> str:
        """
        Get the sample submission file name for the given competition.

        This is used to determine the file name for the submission file.
        """
        input_dir = Path(f"{DS_RD_SETTING.local_data_path}/{competition}")
        sample_submission_files = (
            list(input_dir.glob("*sample_submission*.csv"))
            + list(input_dir.glob("*sampleSubmission*.csv"))
            + list(input_dir.glob("*randomPredictions*.tsv"))
        )
        if len(sample_submission_files) == 0:
            return None
        else:
            return sample_submission_files[0].name

    @abstractmethod
    def is_sub_enabled(self, competition: str) -> bool:
        """
        Is submission file enabled

        If a file like <sample submission csv> is provided; then we think inference from test data to submission file is enabled.
        According test will be enabled as well.

        Why do not we merge `is_sub_enabled` and `enabled`, cases:
        1. The dataset provide evaluation.  But we don't provide submission sample(llm will decide by himself)
        2. We proivde a sample submission. But we don't proivde strict evaluation.

        """
        return self.get_sample_submission_name(competition) is not None


class TestEval(TestEvalBase):
    """The most basic version of evaluation for test data"""

    def __init__(self) -> None:
        super().__init__()
        self.env = get_ds_env()

    def eval(self, competition: str, workspace: FBWorkspace) -> str:
        eval_path = Path(f"{DS_RD_SETTING.local_data_path}/{DS_RD_SETTING.eval_sub_dir}/{competition}")
        if not eval_path.exists():
            err_msg = f"No Test Eval provided due to: {eval_path} not found"
            raise NoTestEvalError(err_msg)
        workspace.inject_files(**{"grade.py": (eval_path / "grade.py").read_text()})
        workspace.inject_files(**{"submission_test.csv": (eval_path / "submission_test.csv").read_text()})
        workspace.execute(
            env=self.env,
            entry=f"python grade.py {competition} | tee {SUBMISSION_SCORE_FILE}",
        )
        workspace.inject_files(**{file: workspace.DEL_KEY for file in ["grade.py", "submission_test.csv"]})
        workspace.execute(
            env=self.env,
            entry=f"chmod 777 {SUBMISSION_SCORE_FILE} || true",
        )
        return (workspace.workspace_path / SUBMISSION_SCORE_FILE).read_text()

    def valid(self, competition: str, workspace: FBWorkspace) -> tuple[str, int]:
        eval_path = Path(f"{DS_RD_SETTING.local_data_path}/{DS_RD_SETTING.eval_sub_dir}/{competition}")
        if not eval_path.exists():
            err_msg = f"No Test Eval provided due to: {eval_path} not found"
            raise NoTestEvalError(err_msg)
        workspace.inject_files(**{"submission_format_valid.py": (eval_path / "valid.py").read_text()})
        workspace.inject_files(**{"submission_test.csv": (eval_path / "submission_test.csv").read_text()})
        submission_result = workspace.run(
            env=self.env,
            entry=f"python submission_format_valid.py {competition}",
        )
        workspace.inject_files(
            **{file: workspace.DEL_KEY for file in ["submission_format_valid.py", "submission_test.csv"]}
        )
        workspace.inject_files(
            **{
                SUBMISSION_FORMAT_CHECK_OUTPUT: submission_result.stdout,
            }
        )
        return submission_result.stdout, submission_result.exit_code

    def enabled(self, competition) -> bool:
        return Path(
            f"{DS_RD_SETTING.local_data_path}/{DS_RD_SETTING.eval_sub_dir}/{competition}/submission_test.csv"
        ).exists()


class MLETestEval(TestEvalBase):
    """Evaluation for test data for MLE-Bench competition"""

    def __init__(self) -> None:
        super().__init__()
        # Mount the data directory to make it available inside the container for mlebench
        extra_volumes = {}
        # Check if we have zip_files directory to mount
        zip_files_path = Path(DS_RD_SETTING.local_data_path) / "zip_files"
        if zip_files_path.exists():
            extra_volumes[str(zip_files_path.resolve())] = "/mle/data"
        
        # get_ds_env() already calls prepare(), so just assign the result
        self.env = get_ds_env(
            conf_type="mlebench",
            extra_volumes=extra_volumes,
        )

    def eval(self, competition: str, workspace: FBWorkspace) -> str:
        workspace.execute(
            env=self.env,
            entry=f"mlebench grade-sample submission.csv {competition} --data-dir /mle/data 2>&1 | tee {SUBMISSION_SCORE_FILE}",
            # NOTE: mlebench does not give output to stdout. so 2>&1 is very necessary !!!!!!
        )
        workspace.execute(
            env=self.env,
            entry=f"chmod 777 {SUBMISSION_SCORE_FILE} || true",
        )
        return (workspace.workspace_path / SUBMISSION_SCORE_FILE).read_text()

    def valid(self, competition: str, workspace: FBWorkspace) -> tuple[str, int]:
        mle_check_code = (
            (Path(__file__).absolute().resolve().parent / "eval_tests" / "mle_submission_format_test.txt")
            .read_text()
            .replace("<competition_id>", competition)
        )
        workspace.inject_files(**{"test/mle_submission_format_test.py": mle_check_code})
        submission_result = workspace.run(env=self.env, entry="python test/mle_submission_format_test.py")

        workspace.inject_files(
            **{
                SUBMISSION_FORMAT_CHECK_OUTPUT: submission_result.stdout,
            }
        )
        return submission_result.stdout, submission_result.exit_code

    def enabled(self, competition) -> bool:
        return True


class KaggleLiveTestEval(TestEvalBase):
    """Evaluate by submitting submission.csv to Kaggle and polling the returned leaderboard score."""

    def __init__(self) -> None:
        super().__init__()
        self.poll_seconds = max(5, int(DS_RD_SETTING.kaggle_submission_poll_seconds))
        self.timeout_seconds = max(self.poll_seconds, int(DS_RD_SETTING.kaggle_submission_timeout_seconds))
        self.complete_without_score_grace_polls = 3
        self._api = None

    def _get_api(self):
        if self._api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi

            self._api = KaggleApi()
            self._api.authenticate()
        return self._api

    @staticmethod
    def _to_float(value: object) -> float | None:
        try:
            if value is None:
                return None
            if isinstance(value, str):
                value = value.strip().replace(",", "")
                if value.endswith("%"):
                    value = value[:-1]
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _submission_to_dict(submission: object) -> dict:
        if submission is None:
            return {}
        if isinstance(submission, dict):
            return submission
        if hasattr(submission, "to_dict"):
            try:
                data = submission.to_dict()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        try:
            return dict(vars(submission))
        except Exception:
            return {}

    def _extract_score(self, submission: object) -> float | None:
        for attr in (
            "publicScore",
            "privateScore",
            "score",
            "public_score",
            "private_score",
            "displayScore",
            "display_score",
            "publicLeaderboardScore",
            "privateLeaderboardScore",
        ):
            score = self._to_float(getattr(submission, attr, None))
            if score is not None:
                return score

        data = self._submission_to_dict(submission)
        for key in (
            "publicScore",
            "privateScore",
            "score",
            "public_score",
            "private_score",
            "displayScore",
            "display_score",
            "publicLeaderboardScore",
            "privateLeaderboardScore",
        ):
            score = self._to_float(data.get(key))
            if score is not None:
                return score

        # Last-resort heuristic for unknown Kaggle schema variants.
        for key, value in data.items():
            if not isinstance(key, str):
                continue
            low = key.lower()
            if "score" not in low:
                continue
            if any(excluded in low for excluded in ("best", "worst", "max", "min", "threshold", "rank")):
                continue
            score = self._to_float(value)
            if score is not None:
                return score
        return None

    def _refresh_submission_by_ref(self, competition: str, submission: object) -> object:
        submission_ref = getattr(submission, "ref", None)
        if not submission_ref:
            return submission

        try:
            submissions = self._get_api().competition_submissions(competition)
        except Exception:
            return submission

        return next((s for s in submissions if getattr(s, "ref", None) == submission_ref), submission)

    def _score_meets_threshold(self, score: float | None, threshold: float | None, is_lower_better: bool | None) -> bool:
        if score is None or threshold is None or is_lower_better is None:
            return False
        if is_lower_better:
            return score <= threshold
        return score >= threshold

    def _extract_leaderboard_scores(self, leaderboard_obj: object) -> list[float]:
        # Kaggle API response formats vary across versions/endpoints; use tolerant extraction.
        rows = []
        for attr in (
            "submissions",
            "leaderboard",
            "entries",
            "teams",
            "leaderboardEntries",
            "rankings",
        ):
            value = getattr(leaderboard_obj, attr, None)
            if isinstance(value, list):
                rows = value
                break

        if not rows and isinstance(leaderboard_obj, list):
            rows = leaderboard_obj

        scores: list[float] = []
        for row in rows:
            score: float | None = None
            for attr in ("score", "publicScore", "privateScore", "displayScore"):
                score = self._to_float(getattr(row, attr, None))
                if score is not None:
                    break

            if score is None and isinstance(row, dict):
                for key in ("score", "publicScore", "privateScore", "displayScore"):
                    score = self._to_float(row.get(key))
                    if score is not None:
                        break

            if score is not None:
                scores.append(score)

        return scores

    def _get_percentile_threshold(self, scores: list[float], percentile: float, is_lower_better: bool) -> float | None:
        if not scores:
            return None
        ordered = sorted(scores, reverse=not is_lower_better)
        index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1))
        return ordered[index]

    def _fetch_leaderboard_thresholds(self, competition: str, is_lower_better: bool | None) -> dict[str, float | None]:
        fallback = {
            "gold_threshold": None,
            "silver_threshold": None,
            "bronze_threshold": None,
            "median_threshold": None,
        }
        if is_lower_better is None:
            return fallback

        api = self._get_api()
        leaderboard = None
        # Prefer official leaderboard view endpoint when available.
        try:
            leaderboard = api.competition_leaderboard_view(competition)
        except Exception as e:
            logger.info(f"Kaggle leaderboard view unavailable for {competition}: {e}")
            return fallback

        scores = self._extract_leaderboard_scores(leaderboard)
        if not scores:
            return fallback

        return {
            "gold_threshold": self._get_percentile_threshold(scores, 0.10, is_lower_better),
            "silver_threshold": self._get_percentile_threshold(scores, 0.20, is_lower_better),
            "bronze_threshold": self._get_percentile_threshold(scores, 0.30, is_lower_better),
            "median_threshold": self._get_percentile_threshold(scores, 0.50, is_lower_better),
        }

    def _wait_submission(self, competition: str, message: str):
        api = self._get_api()
        deadline = time.time() + self.timeout_seconds
        start_time = time.time()
        attempt = 0
        last_status = ""
        complete_without_score_polls = 0
        logger.info(
            f"Waiting for Kaggle scoring: competition={competition}, poll={self.poll_seconds}s, timeout={self.timeout_seconds}s"
        )

        while time.time() < deadline:
            attempt += 1
            submissions = api.competition_submissions(competition)
            current = next((s for s in submissions if str(getattr(s, "description", "")) == message), None)
            if current is not None:
                status = str(getattr(current, "status", "")).lower()
                last_status = status
                elapsed = int(time.time() - start_time)
                logger.info(
                    f"Kaggle submission poll #{attempt}: status='{status or 'unknown'}', elapsed={elapsed}s"
                )
                if self._extract_score(current) is not None:
                    return current
                if "complete" in status:
                    refreshed = self._refresh_submission_by_ref(competition, current)
                    if self._extract_score(refreshed) is not None:
                        return refreshed
                    complete_without_score_polls += 1
                    if complete_without_score_polls >= self.complete_without_score_grace_polls:
                        logger.warning(
                            "Kaggle status is complete but score is still unavailable after "
                            f"{complete_without_score_polls} poll(s); proceeding without score."
                        )
                        return current
                if any(flag in status for flag in ("error", "fail", "invalid", "timeout")):
                    return current
            else:
                elapsed = int(time.time() - start_time)
                logger.info(
                    f"Kaggle submission poll #{attempt}: submission record not visible yet, elapsed={elapsed}s"
                )
            time.sleep(self.poll_seconds)

        raise TimeoutError(
            f"Timed out after {self.timeout_seconds}s waiting for Kaggle submission result for {competition}. "
            f"Last known status: '{last_status or 'unknown'}'"
        )

    def _build_payload(self, competition: str, submission: object) -> dict:
        score = self._extract_score(submission)
        status = str(getattr(submission, "status", ""))

        try:
            is_lower_better = not bool(get_metric_direction(competition))
        except Exception:
            is_lower_better = None

        thresholds = self._fetch_leaderboard_thresholds(competition, is_lower_better)
        above_median = self._score_meets_threshold(score, thresholds["median_threshold"], is_lower_better)
        gold_medal = self._score_meets_threshold(score, thresholds["gold_threshold"], is_lower_better)
        silver_medal = self._score_meets_threshold(score, thresholds["silver_threshold"], is_lower_better)
        bronze_medal = self._score_meets_threshold(score, thresholds["bronze_threshold"], is_lower_better)

        created_at = getattr(submission, "date", None)
        if hasattr(created_at, "isoformat"):
            created_at = created_at.isoformat()
        elif created_at is not None:
            created_at = str(created_at)

        valid_submission = bool(score is not None and "error" not in status.lower())
        return {
            "competition_id": competition,
            "score": score,
            "gold_threshold": thresholds["gold_threshold"],
            "silver_threshold": thresholds["silver_threshold"],
            "bronze_threshold": thresholds["bronze_threshold"],
            "median_threshold": thresholds["median_threshold"],
            "any_medal": bool(gold_medal or silver_medal or bronze_medal),
            "gold_medal": gold_medal,
            "silver_medal": silver_medal,
            "bronze_medal": bronze_medal,
            "above_median": above_median,
            "submission_exists": True,
            "valid_submission": valid_submission,
            "is_lower_better": is_lower_better,
            "created_at": created_at,
            "submission_path": "submission.csv",
            "reference_based": False,
            "reference_type": "kaggle_live_api",
            "kaggle_status": status,
            "kaggle_ref": getattr(submission, "ref", None),
        }

    def _emit_format_check_output(self, workspace: FBWorkspace, message: str) -> None:
        workspace.inject_files(**{SUBMISSION_FORMAT_CHECK_OUTPUT: message})

    def eval(self, competition: str, workspace: FBWorkspace) -> str:
        submission_path = workspace.workspace_path / "submission.csv"
        if not submission_path.exists():
            raise NoTestEvalError(f"submission.csv not found in workspace: {workspace.workspace_path}")

        api = self._get_api()
        message = f"rdagent-live-{workspace.workspace_path.name}-{uuid.uuid4().hex[:8]}"
        api.competition_submit(str(submission_path), message, competition)
        submission = self._wait_submission(competition, message)
        payload = self._build_payload(competition, submission)
        payload_str = json.dumps(payload)
        (workspace.workspace_path / SUBMISSION_SCORE_FILE).write_text(payload_str)
        return payload_str

    def valid(self, competition: str, workspace: FBWorkspace) -> tuple[str, int]:
        submission_path = workspace.workspace_path / "submission.csv"
        if not submission_path.exists():
            msg = "submission.csv not found"
            self._emit_format_check_output(workspace, msg)
            return msg, 1

        sample_name = self.get_sample_submission_name(competition)
        if sample_name is None:
            msg = "sample submission file not found under local competition folder"
            self._emit_format_check_output(workspace, msg)
            return msg, 1

        sample_path = Path(f"{DS_RD_SETTING.local_data_path}/{competition}/{sample_name}")
        if not sample_path.exists():
            msg = f"sample submission file not found: {sample_path}"
            self._emit_format_check_output(workspace, msg)
            return msg, 1

        sample_df = pd.read_csv(sample_path)
        submission_df = pd.read_csv(submission_path)

        if submission_df.columns.tolist() != sample_df.columns.tolist():
            msg = f"submission columns mismatch, expected={sample_df.columns.tolist()}, got={submission_df.columns.tolist()}"
            self._emit_format_check_output(workspace, msg)
            return msg, 1
        if len(submission_df) != len(sample_df):
            msg = f"submission row count mismatch, expected={len(sample_df)}, got={len(submission_df)}"
            self._emit_format_check_output(workspace, msg)
            return msg, 1

        if not submission_df.iloc[:, 0].equals(sample_df.iloc[:, 0]):
            msg = "submission id column order/content mismatch with sample submission"
            self._emit_format_check_output(workspace, msg)
            return msg, 1

        msg = "Submission format is valid for Kaggle live submission."
        self._emit_format_check_output(workspace, msg)
        return msg, 0

    def enabled(self, competition) -> bool:
        try:
            self._get_api().competition_submissions(competition)
            return self.get_sample_submission_name(competition) is not None
        except Exception:
            return False


def get_test_eval() -> TestEvalBase:
    """Get the test evaluation instance"""
    if DS_RD_SETTING.kaggle_live_eval:
        return KaggleLiveTestEval()
    if DS_RD_SETTING.if_using_mle_data:
        return MLETestEval()
    return TestEval()
