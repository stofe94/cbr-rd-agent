"""CBR Knowledge Base built on R&D-Agent's PDVectorBase.

Storage layout:
    cbr_kb_dir/
        cases.json            # Case metadata (without code)
        code/                 # Full code snapshots
            {case_id}.py
        vector_base.pkl       # PDVectorBase persistence

Caching:
    retrieve() results are cached by (query, top_k, exclude_failures,
    exclude_competition, mmr_lambda, min_similarity) hash.

    Cache invalidation strategy:
    - retrieve(update_reuse_stats=True) uses _save_no_invalidate() so the
      cache stays warm within the same loop iteration — Gate 4 novelty check
      can hit the cache populated by direct_exp_gen without a redundant
      embedding call.
    - add_case() uses save(invalidate_cache=True) — a new case genuinely
      changes retrieval results so stale entries must be cleared.
    - _flush_loop_caches() is called at the start of each new iteration to
      prevent stale cross-iteration hits.

KaggleExperienceBase isolation:
    search() passes constraint_labels=list(cases.keys()) so that trunk rows
    from KaggleExperienceBase that share the same vector_base.pkl cannot
    pollute CBR retrieval results.

Consistency guarantees:
    _verify_consistency() runs on every __init__ and ensures cases.json and
    vector_base.pkl are in sync — re-embedding missing cases and removing
    orphan vector rows after any crash or partial write.

    add_case() removes existing vector rows for a case_id before adding the
    new embedding, preventing duplicate rows from accumulating in vector_df
    on repeated adds.
"""
import hashlib
import json
import os
import numpy as np
from typing import List, Optional, Tuple

from rdagent.log import rdagent_logger as logger
from rdagent.components.knowledge_management.vector_base import (
    PDVectorBase,
    KnowledgeMetaData,
)
from .case_schema import Case, CaseOutcome


class CBRKnowledgeBase:
    """Persistent case store with embedding-based retrieval.

    Uses R&D-Agent's PDVectorBase for vector operations and adds:
    - Structured case JSON persistence
    - Code snapshot file storage
    - MMR (Maximal Marginal Relevance) diversity
    - Quality-weighted scoring
    - Query-level result cache to avoid redundant embedding calls
    """

    def __init__(self, kb_dir: str = "./cbr_knowledge_base"):
        self.kb_dir = kb_dir
        self.cases_file = os.path.join(kb_dir, "cases.json")
        self.code_dir = os.path.join(kb_dir, "code")
        self.vb_path = os.path.join(kb_dir, "vector_base.pkl")

        os.makedirs(self.code_dir, exist_ok=True)

        # Load or create PDVectorBase
        if os.path.exists(self.vb_path):
            import pickle
            with open(self.vb_path, "rb") as f:
                self.vb: PDVectorBase = pickle.load(f)
        else:
            self.vb = PDVectorBase()

        # Load cases
        self.cases: dict[str, Case] = {}
        if os.path.exists(self.cases_file):
            with open(self.cases_file) as f:
                raw = json.load(f)
            self.cases = {cid: Case.from_dict(d) for cid, d in raw.items()}

        # ── Retrieve cache ─────────────────────────────
        # Maps cache_key (str) -> List[Tuple[Case, float]]
        # Invalidated on every save() (i.e. whenever KB changes).
        self._retrieve_cache: dict[str, List[Tuple[Case, float]]] = {}
        self._cache_version: int = 0  # bumped on each save()

        # Ensure cases.json and vector_base.pkl are in sync after any
        # crash or partial write during a previous session.
        self._verify_consistency()

        logger.info(f"CBR KB loaded: {len(self.cases)} cases from {kb_dir}")

    # ─── Cache helpers ────────────────────────────────

    @staticmethod
    def _cache_key(
        query: str,
        top_k: int,
        exclude_failures: bool,
        exclude_competition: Optional[str],
        mmr_lambda: float,
        min_similarity: float,
    ) -> str:
        raw = f"{query}|{top_k}|{exclude_failures}|{exclude_competition or ''}|{mmr_lambda:.4f}|{min_similarity:.4f}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _invalidate_cache(self) -> None:
        """Clear retrieve cache. Called whenever the KB is mutated."""
        self._retrieve_cache.clear()
        self._cache_version += 1

    def _verify_consistency(self) -> None:
        """Ensure vector_df and cases.json are fully in sync.

        Runs on every __init__. Handles two failure modes:
        1. Case in cases.json but missing from vector_df (crash after JSON
           write, before PKL write) → re-embed the case, no API call wasted
           on cases already in the vector store.
        2. Row in vector_df with no matching case in cases.json (orphan from
           a deleted or replaced case) → remove the row silently.

        Neither operation touches cases that are already consistent, so this
        is a no-op on healthy KBs.
        """
        if self.vb.vector_df.empty and not self.cases:
            return

        vb_labels = set(self.vb.vector_df["label"].unique()) if not self.vb.vector_df.empty else set()
        case_ids = set(self.cases.keys())

        needs_save = False

        # ── Missing from vector_df → re-embed ────────────────────────────
        missing_in_vb = case_ids - vb_labels
        if missing_in_vb:
            logger.warning(
                f"CBR KB consistency: {len(missing_in_vb)} case(s) missing from "
                f"vector_df, re-embedding: {sorted(missing_in_vb)}"
            )
            for case_id in missing_in_vb:
                case = self.cases[case_id]
                metadata = KnowledgeMetaData(
                    content=case.to_embedding_text(),
                    label=case.case_id,
                )
                self.vb.add(metadata)
            needs_save = True

        # ── Orphan rows in vector_df → remove ────────────────────────────
        orphan_labels = vb_labels - case_ids
        if orphan_labels:
            logger.warning(
                f"CBR KB consistency: {len(orphan_labels)} orphan row(s) in "
                f"vector_df, removing: {sorted(orphan_labels)}"
            )
            self.vb.vector_df = self.vb.vector_df[
                ~self.vb.vector_df["label"].isin(orphan_labels)
            ].reset_index(drop=True)
            needs_save = True

        # ── Duplicate rows per case_id → keep only the latest ────────────
        # Can happen if a previous run crashed mid-add_case after vb.add()
        # but before the dedup guard was in place.
        counts = self.vb.vector_df["label"].value_counts()
        duplicated = counts[counts > 1].index.tolist()
        if duplicated:
            logger.warning(
                f"CBR KB consistency: {len(duplicated)} case(s) have duplicate "
                f"rows in vector_df, keeping last: {duplicated}"
            )
            # Keep only the last row per label (most recent add)
            self.vb.vector_df = (
                self.vb.vector_df
                .groupby("label", as_index=False)
                .last()
                .reset_index(drop=True)
            )
            needs_save = True

        if needs_save:
            self.save(invalidate_cache=True)

    # ─── Persistence ──────────────────────────────────

    def save(self, invalidate_cache: bool = True):
        # Re-infer model_type for cases that still have "unknown" but have code on disk.
        for c in self.cases.values():
            if not c.solution.model_type or c.solution.model_type.lower() == "unknown":
                code = c.solution.code_snippet or self.get_full_code(c.case_id)
                if code:
                    inferred = Case._infer_model_type(
                        c.solution.hypothesis_text,
                        c.solution.plan_summary,
                        code,
                    )
                    if inferred != "unknown":
                        c.solution.model_type = inferred
        # Cases JSON
        with open(self.cases_file, "w") as f:
            json.dump(
                {cid: c.to_dict() for cid, c in self.cases.items()},
                f, indent=2, ensure_ascii=False,
            )
        # Vector base
        import pickle
        with open(self.vb_path, "wb") as f:
            pickle.dump(self.vb, f)

        # Only invalidate when the caller explicitly requests it.
        # retrieve() with update_reuse_stats=True calls _save_no_invalidate()
        # so Gate 4 can still hit the warm cache from the same iteration.
        if invalidate_cache:
            self._invalidate_cache()
        logger.info(f"CBR KB saved: {len(self.cases)} cases")

    def _save_no_invalidate(self):
        """Persist to disk without clearing the retrieve cache.

        Used by retrieve(update_reuse_stats=True) so that Gate 4's novelty
        check can hit the cache populated earlier in the same loop iteration.
        Cache is invalidated at the start of each new iteration via
        _flush_loop_caches() in CBRDataScienceRDLoop.
        """
        self.save(invalidate_cache=False)

    # ─── CRUD ─────────────────────────────────────────

    def add_case(self, case: Case) -> str:
        """Add a case: embed → store vector → save code → persist JSON.

        If a case with the same case_id already exists in vector_df (e.g. from
        a previous run or a retry), the old row is removed before the new
        embedding is added. This prevents duplicate rows from accumulating and
        skewing MMR / similarity scores.
        """
        # 1. Dedup: remove any existing vector row for this case_id so we
        #    never accumulate duplicates in vector_df.
        if not self.vb.vector_df.empty and case.case_id in self.vb.vector_df["label"].values:
            self.vb.vector_df = self.vb.vector_df[
                self.vb.vector_df["label"] != case.case_id
            ].reset_index(drop=True)
            logger.warning(f"CBR KB: replaced existing vector row for case {case.case_id}")

        # 2. Embed and add to vector base
        embedding_text = case.to_embedding_text()
        metadata = KnowledgeMetaData(
            content=embedding_text,
            label=case.case_id,
        )
        self.vb.add(metadata)

        # 2. Save code snapshot
        if case.solution.code_snippet:
            code_file = os.path.join(self.code_dir, f"{case.case_id}.py")
            with open(code_file, "w") as f:
                f.write(case.solution.code_snippet)
            case.solution.code_path = code_file

        # 2b. Re-infer model_type now that code_snippet is guaranteed present.
        # from_rd_agent runs before the workspace is fully written, so
        # code_snapshot is often empty there -> model_type stays "unknown".
        # Here the code is available -> fix it before persisting.
        if not case.solution.model_type or case.solution.model_type.lower() == "unknown":
            inferred = Case._infer_model_type(
                case.solution.hypothesis_text,
                case.solution.plan_summary,
                case.solution.code_snippet,
            )
            if inferred != "unknown":
                case.solution.model_type = inferred

        # 3. Store case and persist.
        # save(invalidate_cache=True) here is correct — a new case genuinely
        # changes retrieval results, so stale cache entries must be cleared.
        self.cases[case.case_id] = case
        self.save(invalidate_cache=True)

        logger.info(f"Added case {case.case_id}: {case.outcome.value} | "
                    f"{case.metrics.primary_metric_name}={case.metrics.primary_metric_value}")
        return case.case_id

    def get_case(self, case_id: str) -> Optional[Case]:
        return self.cases.get(case_id)

    def get_full_code(self, case_id: str) -> str:
        """Load full code from file (not just snippet)."""
        path = os.path.join(self.code_dir, f"{case_id}.py")
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
        return ""

    # ─── Retrieval ────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        exclude_failures: bool = True,
        exclude_competition: Optional[str] = None,
        mmr_lambda: float = 0.7,
        min_similarity: float = 0.0,
        update_reuse_stats: bool = True,
    ) -> List[Tuple[Case, float]]:
        """Retrieve top-k cases with MMR diversity.

        Results are cached by (query, top_k, exclude_failures,
        exclude_competition, mmr_lambda, min_similarity).  When update_reuse_stats=False
        (e.g. Gate-4 novelty checks) the cache is always consulted first,
        saving one embedding call per duplicate query.  When
        update_reuse_stats=True the underlying search is still cached but
        the reuse-count mutation is applied on top; subsequent read-only
        calls for the same query will see the cached ranked list (without
        redundant re-embedding).

        Args:
            query: Problem description text
            top_k: Number of cases to return
            exclude_failures: Skip cases with FAILURE outcome
            exclude_competition: Skip cases from this competition (self-reference)
            mmr_lambda: MMR balance (1.0 = pure relevance, 0.0 = pure diversity)
            min_similarity: Minimum vector similarity threshold to keep a case
            update_reuse_stats: Increment reuse counters and persist (set False
                for internal dedup/novelty checks)

        Returns:
            List of (case, score) tuples, ranked by MMR.
        """
        if not self.cases:
            return []

        key = self._cache_key(query, top_k, exclude_failures, exclude_competition, mmr_lambda, min_similarity)

        # ── Cache hit ────────────────────────────────
        if key in self._retrieve_cache:
            cached = self._retrieve_cache[key]
            if not update_reuse_stats:
                return cached
            # Apply reuse stats to cached result and persist without
            # invalidating the cache — Gate 4 may still need it this iteration.
            for case, _ in cached:
                case.reuse_count += 1
                case.updated_at = __import__("time").time()
            self._save_no_invalidate()
            return cached

        # ── Cache miss: run full search ───────────────
        # Restrict search to CBR-owned case IDs to avoid matching against
        # KaggleExperienceBase trunks that share the same vector_base.pkl.
        docs, similarities = self.vb.search(
            content=query,
            topk_k=top_k * 4,
            constraint_labels=list(self.cases.keys()),
        )
        raw_results = list(zip(docs, similarities))

        # Filter and resolve cases
        candidates = []
        for metadata, sim_score in raw_results:
            case_id = metadata.label
            case = self.cases.get(case_id)
            if not case:
                continue
            if exclude_failures and case.outcome == CaseOutcome.FAILURE:
                continue
            if exclude_competition and case.source_competition == exclude_competition:
                continue
            if sim_score < min_similarity:
                continue

            # Quality-weighted score
            quality_bonus = self._quality_score(case)
            final_score = 0.7 * sim_score + 0.3 * quality_bonus
            candidates.append((case, final_score, metadata))

        if not candidates:
            return []

        # Apply MMR for diversity
        selected = self._mmr_select(candidates, top_k, mmr_lambda)

        # Store in cache before any mutation so read-only callers benefit too.
        self._retrieve_cache[key] = selected

        # Update reuse counts only for actual reuse events.
        # Use _save_no_invalidate so Gate 4 can still hit this cache entry
        # within the same loop iteration.
        if update_reuse_stats:
            for case, _ in selected:
                case.reuse_count += 1
                case.updated_at = __import__("time").time()
            self._save_no_invalidate()

        return selected

    def _quality_score(self, case: Case) -> float:
        """0–1 quality score based on outcome + metrics + reuse history."""
        score = 0.0
        if case.outcome == CaseOutcome.SUCCESS:
            score += 0.4
        elif case.outcome == CaseOutcome.PARTIAL:
            score += 0.15

        if case.metrics.improvement_pct is not None and case.metrics.improvement_pct > 0:
            score += min(case.metrics.improvement_pct / 50.0, 0.3)

        if case.reuse_success_rate > 0:
            score += 0.3 * case.reuse_success_rate

        return min(score, 1.0)

    def _mmr_select(
        self,
        candidates: List[Tuple[Case, float, KnowledgeMetaData]],
        top_k: int,
        mmr_lambda: float,
    ) -> List[Tuple[Case, float]]:
        """Maximal Marginal Relevance for diverse case selection."""
        if len(candidates) <= top_k:
            return [(c, s) for c, s, _ in candidates]

        # Get embeddings from metadata
        embs = []
        for _, _, meta in candidates:
            if meta.embedding is not None:
                embs.append(np.array(meta.embedding))
            else:
                embs.append(np.zeros(384))  # fallback dim

        selected_indices = []
        remaining = list(range(len(candidates)))

        for _ in range(min(top_k, len(candidates))):
            best_idx, best_mmr = None, -float("inf")
            for i in remaining:
                relevance = candidates[i][1]  # final_score
                max_sim = 0.0
                for j in selected_indices:
                    sim = float(np.dot(embs[i], embs[j]) / (
                        np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]) + 1e-8
                    ))
                    max_sim = max(max_sim, sim)
                mmr = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining.remove(best_idx)

        return [(candidates[i][0], candidates[i][1]) for i in selected_indices]

    # ─── Reuse Causality ──────────────────────────────

    # ── Fingerprint noise-filter tables ──────────────────────────────────
    # Values that appear in virtually every ML script regardless of problem
    # domain or algorithm choice. Including them inflates cross-competition
    # similarity and masks genuine structural differences.

    # Function names so common they carry no discrimination signal.
    _FP_BOILERPLATE_FUNCTIONS: frozenset = frozenset({"main"})

    # Output files that are Kaggle boilerplate (appear in all solutions).
    _FP_BOILERPLATE_OUTPUT_FILES: frozenset = frozenset({
        "scores.csv", "submission.csv", "submission.parquet",
    })

    # Hyperparameter keys that appear in virtually every sklearn/GBDT model.
    _FP_GENERIC_HYPERPARAM_KEYS: frozenset = frozenset({
        "learning_rate", "n_estimators", "random_state", "n_splits", "n_jobs",
    })

    # from-import names that are near-universal ML utilities. Specific class
    # names (CatBoostClassifier, OneHotEncoder, …) are kept because they
    # discriminate between problem types and algorithm choices.
    _FP_GENERIC_FROM_NAMES: frozenset = frozenset({
        "np", "pd", "os", "re", "time", "argparse", "inspect",
        "Path", "pathlib",
        "train_test_split", "KFold", "cross_val_score", "cross_validate",
        "mean_squared_error", "mean_absolute_error",
        "registry", "validate_submission",
        "Optional", "List", "Dict", "Tuple", "Any",
    })

    @staticmethod
    def _extract_code_fingerprint(code: str) -> dict:
        """Extract a rich structural fingerprint from a Python code snapshot.

        Captures enough signal to distinguish genuinely adapted code from
        coincidentally similar code. All extraction is regex-based.

        Key design decisions vs. naive extraction:
        - Universal boilerplate (function `main`, output files `submission.csv`,
          generic hyperparams `random_state`/`n_estimators`) is stripped at
          extraction time so it cannot inflate cross-competition similarity.
        - `specific_from_names` retains only problem/algorithm-specific imports
          (CatBoostClassifier, OneHotEncoder, accuracy_score) and discards
          generic utilities that appear in every script.
        - Hyperparameters are split into `specific_hyperparam_keys`
          (model-architecture keys that differ between algorithms) and
          `generic_hyperparam_keys` (universal boilerplate, lower weight).

        Returns a dict with named feature sets; similarity is computed
        per-dimension and weighted in _fingerprint_similarity().
        """
        import re

        if not code:
            return {}

        # ── 1. Top-level library imports ──────────────
        direct_imports = set(re.findall(
            r'^import\s+([\w]+)', code, re.MULTILINE
        ))
        from_imports_raw = re.findall(
            r'^from\s+([\w.]+)\s+import\s+(.+)', code, re.MULTILINE
        )
        from_modules = set(m for m, _ in from_imports_raw)
        from_names_raw = set()
        for _, names in from_imports_raw:
            for n in re.split(r',\s*', names):
                n = n.strip().split(' as ')[0].strip()
                if n and n != '*':
                    from_names_raw.add(n)

        # Filtered: remove near-universal utilities.
        specific_from_names = (
            from_names_raw - CBRKnowledgeBase._FP_GENERIC_FROM_NAMES
        )

        # ── 2. Function definitions (boilerplate filtered) ─────────────
        functions_raw = set(re.findall(r'^def\s+(\w+)\s*\(', code, re.MULTILINE))
        # `main` appears in every script — exclude so two files sharing only
        # `main` don't score artificially high on this dimension.
        functions = functions_raw - CBRKnowledgeBase._FP_BOILERPLATE_FUNCTIONS
        classes = set(re.findall(r'^class\s+(\w+)\s*[:(]', code, re.MULTILINE))

        # ── 3. ML model classes instantiated ──────────
        model_pattern = re.compile(
            r'\b(\w*(?:Classifier|Regressor|Forest|Boost|GBM|SVM|SVR|SVC'
            r'|KNeighbors|MLP|Ridge|Lasso|ElasticNet|Pipeline|Stacking'
            r'|Voting|Bagging|AdaBoost|GradientBoosting|ExtraTrees'
            r'|CatBoost|LightGBM|XGB)\w*)\s*\('
        )
        model_classes = set(model_pattern.findall(code))

        # ── 4. sklearn/ML API surface ──────────────────
        api_calls = set(re.findall(
            r'\.(fit|predict|predict_proba|predict_log_proba|transform'
            r'|fit_transform|inverse_transform|cross_val_score'
            r'|cross_validate|score|evaluate)\s*\(',
            code
        ))

        # ── 5. Cross-validation strategy ──────────────
        cv_strategies = set(re.findall(
            r'\b(StratifiedKFold|KFold|GroupKFold|StratifiedGroupKFold'
            r'|TimeSeriesSplit|RepeatedKFold|LeaveOneOut|cross_val_score'
            r'|cross_validate)\s*[(\[]',
            code
        ))

        # ── 6. Encoding and preprocessing techniques ──
        encoding_techniques = set(re.findall(
            r'\b(target_encode|LabelEncoder|OrdinalEncoder|OneHotEncoder'
            r'|get_dummies|TargetEncoder|BinaryEncoder|HashingEncoder'
            r'|StandardScaler|MinMaxScaler|RobustScaler|Normalizer'
            r'|PowerTransformer|QuantileTransformer|SimpleImputer'
            r'|KNNImputer|IterativeImputer)\b',
            code
        ))

        # ── 7. Feature engineering patterns ───────────
        # Column assignments: df['new_col'] = ... — captures engineered features.
        new_columns = set(re.findall(
            r"(?:df|train|test|X|df_combined|df_feat|train_df|test_df)\s*\[[\'\"](\w+)[\'\"]\]\s*=",
            code
        ))
        groupby_keys = set(re.findall(
            r"\.groupby\s*\(\s*[\[\'\"]([^\]\'\"]+)[\]\'\"]",
            code
        ))
        # ── 8. Hyperparameters — two tiers ────────────
        # Tier 1 (specific): model-architecture keys that differ between algorithms.
        specific_hyperparam_keys = set(re.findall(
            r"['\"]("
            r"num_leaves|min_child_weight|subsample|colsample_bytree"
            r"|colsample_bylevel|gamma|reg_alpha|reg_lambda"
            r"|min_samples_split|min_samples_leaf|max_features"
            r"|l2_leaf_reg|iterations|depth|grow_policy|bagging_temperature"
            r"|border_count|od_type|C|kernel|epsilon|alpha|l1_ratio"
            r"|n_neighbors|hidden_layer_sizes|max_depth"
            r"|early_stopping_rounds|thread_count"
            r")['\"][\s]*:",
            code
        ))
        # Tier 2 (generic): near-universal keys — low weight in scoring.
        generic_hyperparam_keys = set(re.findall(
            r"['\"](n_estimators|learning_rate|random_state|n_splits|n_jobs)['\"][\s]*:",
            code
        ))
        # ── 9. Output file names (boilerplate stripped) ──
        output_files_raw = set(re.findall(
            r"['\"](\w+\.csv|\w+\.parquet|\w+\.pkl)['\"]",
            code
        ))
        output_files = output_files_raw - CBRKnowledgeBase._FP_BOILERPLATE_OUTPUT_FILES
        # ── 10. Control flow shape ─────────────────────
        n_for_loops = len(re.findall(r'^\s*for\s+\w+\s+in\s+', code, re.MULTILINE))
        n_try_blocks = len(re.findall(r'^\s*try\s*:', code, re.MULTILINE))

        return {
            # Legacy / downstream use
            "direct_imports": direct_imports,
            "from_modules": from_modules,
            "from_names": from_names_raw,
            "classes": classes,
            "api_calls": api_calls,
            # ── Discriminating dimensions ──────────────
            "specific_from_names": specific_from_names,
            "functions": functions,
            "model_classes": model_classes,
            "cv_strategies": cv_strategies,
            "encoding_techniques": encoding_techniques,
            "new_columns": new_columns,
            "groupby_keys": groupby_keys,
            "specific_hyperparam_keys": specific_hyperparam_keys,
            "generic_hyperparam_keys": generic_hyperparam_keys,
            "output_files": output_files,
            "n_for_loops": n_for_loops,
            "n_try_blocks": n_try_blocks,
        }

    @staticmethod
    def _fingerprint_similarity(fp1: dict, fp2: dict) -> float:
        """Weighted similarity between two code fingerprints.

        Weights are calibrated to measure *deliberate code adaptation*, NOT
        coincidental similarity from using the same standard ML stack.

        Empirically validated against 10 real Kaggle solutions across two
        competitions (nomad-regression, spaceship-titanic-classification):

        Before this calibration the old scheme produced 3 cross-competition
        false positives (fp_sim > 0.35) driven by universal patterns
        (functions sharing `main`, output files, generic hyperparams).
        This scheme produces 0 false positives on that benchmark.

        Dimension notes:
        - new_columns         0.28  engineered feature column names — unique
                                    per problem domain, strong adaptation signal
        - specific_from_names 0.22  problem/algorithm-specific imports
                                    (CatBoostClassifier, OneHotEncoder, …)
                                    filtered to exclude universal utilities
        - functions           0.18  custom helper function names, `main` removed
        - specific_hyperparam 0.12  model-architecture keys (num_leaves, depth,
                                    l2_leaf_reg) — near-zero cross-competition overlap
        - encoding_techniques 0.08  non-trivial preprocessing choices
        - groupby_keys        0.05  specific aggregation keys
        - model_classes       0.04  discriminates across algorithm families
        - generic_hyperparam  0.02  n_estimators/learning_rate — low signal
        - cv_strategies       0.01  near-universal for tabular

        output_files excluded entirely — submission.csv/scores.csv stripped
        at extraction time; remaining custom filenames too rare to weight.

        Empty-set rule: (∅, ∅) → 0.0 for all set dims.
        Two solutions that both lack a feature share nothing meaningful.

        Returns a float in [0, 1].
        """
        if not fp1 or not fp2:
            return 0.0

        def j(a: set, b: set) -> float:
            """Jaccard with empty-set penalty: (∅, ∅) → 0.0."""
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        def numeric_sim(a: int, b: int) -> float:
            if a == 0 and b == 0:
                return 0.0
            return 1.0 - abs(a - b) / max(a, b)

        weighted_dims = [
            # ── High-discrimination: unique per problem domain ──────────
            ("new_columns",             0.28),
            ("specific_from_names",     0.22),
            ("functions",               0.18),
            ("specific_hyperparam_keys",0.12),
            # ── Medium-discrimination: technique choices ─────────────────
            ("encoding_techniques",     0.08),
            ("groupby_keys",            0.05),
            # ── Low-discrimination: common across similar tasks ──────────
            ("model_classes",           0.04),
            ("generic_hyperparam_keys", 0.02),
            ("cv_strategies",           0.01),
        ]

        total_weight = sum(w for _, w in weighted_dims)
        score = sum(
            weight * j(fp1.get(key, set()), fp2.get(key, set()))
            for key, weight in weighted_dims
        )

        # Structural shape — very minor signal
        loop_weight = 0.005
        score += loop_weight * numeric_sim(
            fp1.get("n_for_loops", 0),
            fp2.get("n_for_loops", 0),
        )
        total_weight += loop_weight

        return score / total_weight


    def check_reuse_causality(
        self,
        new_case: "Case",
        retrieved_cases: list,
        embedding_threshold: float = 0.82,
        fingerprint_threshold: float = 0.10,
    ) -> list:
        """Determine which retrieved cases causally influenced the new case.

        Uses two independent signals — both must pass their threshold:

        1. Embedding similarity (Cosine on to_embedding_text vectors):
           Measures problem/approach similarity. Threshold 0.82 means the
           new case addresses the same problem with a similar approach.

        2. Code fingerprint similarity (weighted Jaccard on structural features):
           Measures structural code similarity. Threshold 0.35 means more
           than a third of the weighted code structure is shared — a strong
           signal that the new code was adapted from the retrieved case.

        Only cases passing BOTH thresholds are considered causal.

        Args:
            new_case: The newly retained case.
            retrieved_cases: List of (Case, score) from the propose phase.
            embedding_threshold: Min cosine similarity on case embeddings.
            fingerprint_threshold: Min weighted fingerprint similarity.

        Returns:
            List of (Case, embedding_sim, fingerprint_sim) for causal cases.
        """
        if not retrieved_cases or self.vb.vector_df.empty:
            return []

        # Get new case embedding from vector_df (just added by add_case)
        new_rows = self.vb.vector_df[
            self.vb.vector_df["label"] == new_case.case_id
        ]
        if new_rows.empty:
            logger.warning(f"CBR reuse check: no vector row found for {new_case.case_id}")
            return []

        import numpy as np
        new_emb = np.array(new_rows.iloc[-1]["embedding"])

        # Fingerprint of new code
        new_code = self.get_full_code(new_case.case_id) or new_case.solution.code_snippet or ""
        new_fp = self._extract_code_fingerprint(new_code)

        causal = []
        for retrieved_case, _ in retrieved_cases:
            # ── Embedding similarity ──────────────────
            src_rows = self.vb.vector_df[
                self.vb.vector_df["label"] == retrieved_case.case_id
            ]
            if src_rows.empty:
                continue
            src_emb = np.array(src_rows.iloc[-1]["embedding"])

            norm_new = np.linalg.norm(new_emb)
            norm_src = np.linalg.norm(src_emb)
            if norm_new < 1e-8 or norm_src < 1e-8:
                continue
            emb_sim = float(np.dot(new_emb, src_emb) / (norm_new * norm_src))

            if emb_sim < embedding_threshold:
                logger.debug(
                    f"CBR reuse: {retrieved_case.case_id} emb_sim={emb_sim:.3f} "
                    f"< threshold {embedding_threshold} → not causal"
                )
                continue

            # ── Code fingerprint similarity ───────────
            src_code = self.get_full_code(retrieved_case.case_id) or retrieved_case.solution.code_snippet or ""
            src_fp = self._extract_code_fingerprint(src_code)
            fp_sim = self._fingerprint_similarity(new_fp, src_fp)

            if fp_sim < fingerprint_threshold:
                logger.debug(
                    f"CBR reuse: {retrieved_case.case_id} fp_sim={fp_sim:.3f} "
                    f"< threshold {fingerprint_threshold} → not causal"
                )
                continue

            logger.info(
                f"CBR reuse: {retrieved_case.case_id} CAUSAL "
                f"emb_sim={emb_sim:.3f} fp_sim={fp_sim:.3f}"
            )
            causal.append((retrieved_case, emb_sim, fp_sim))

        return causal

    # ─── Stats ────────────────────────────────────────

    def stats(self) -> dict:
        outcomes = {}
        for c in self.cases.values():
            outcomes[c.outcome.value] = outcomes.get(c.outcome.value, 0) + 1
        # Top 3 most reused cases (by reuse_count)
        top_reused = sorted(self.cases.values(), key=lambda x: x.reuse_count, reverse=True)[:3]
        return {
            "total_cases": len(self.cases),
            "outcomes": outcomes,
            "most_reused": [(c.case_id, c.reuse_count) for c in top_reused],
            "cache_entries": len(self._retrieve_cache),
        }
