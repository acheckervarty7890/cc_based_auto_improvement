"""Balanced, periodically-reshuffled sampling of past attempts for the attacker.

``view_past_attempts`` no longer just returns the most recent rows. Instead a
single :class:`ViewSampler` is shared across the whole rotation (all concurrent
attacker sessions on one JSONL) and decides what the attacker sees:

  * **Balance** — by default the shown set is ~50/50 successful vs. unsuccessful
    attempts, total capped at ``limit``. If one side is short, the remainder is
    filled from the other so the attacker still sees ``limit`` examples.
  * **Training seeds** — true-class examples from the base training set are
    blended into the *successful* pool (tagged ``success=True``) so the attacker
    always has concrete examples of what the true class looks like, even before
    any real successes exist. They stay in the pool after real finds accumulate.
  * **Periodic reshuffle** — when ``reshuffle`` is True (the default), the pool is
    snapshotted and randomly shuffled once per ``reshuffle_interval`` *submissions*
    (counted via the shared store's total row count). Within an interval the draw
    is stable; crossing an interval boundary redraws a fresh random sample. The
    shuffle uses its own seeded RNG, independent of the global RNG, so it's
    reproducible regardless of drift. When ``reshuffle`` is False, no randomness is
    used: the attacker is shown the *most-recent* successful and unsuccessful
    attempts (recency order), and training seeds are used only as a **fallback** for
    the successful half when there are no real successful attempts yet.

Note there is **no judge-confidence filter** here — confidence gating belongs to
the training/postprocessing path (``retrain_probe(min_judge_confidence=...)``),
not to what the attacker is shown.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from agentic_redteam.persistence import AttemptRecord, Conversation, JsonlStore

# attacker_model value used to mark a shown example as a training-set seed rather
# than a real submitted attempt.
SEED_ATTACKER_TAG = "__training_seed__"


def load_true_class_seeds(
    base_training_data_path: str | Path, true_class_label: str
) -> list[Conversation]:
    """Load conversations of the run's *true* class from the base training JSONL.

    The base training rows look like ``{"inputs": <json-string or list of
    {role,content}>, "labels": <human label>, ...}``. We keep only rows whose
    ``labels`` equals ``true_class_label`` (the class a successful find must
    belong to: negative for false_positive, positive for false_negative) and
    return them as :class:`Conversation` objects. Rows that don't parse are
    skipped rather than aborting the run. Non-JSONL inputs yield no seeds.
    """
    path = Path(base_training_data_path)
    seeds: list[Conversation] = []
    if not path.exists() or path.suffix.lower() != ".jsonl":
        return seeds
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if str(row.get("labels")) != true_class_label:
                    continue
                raw = row.get("inputs")
                msgs = json.loads(raw) if isinstance(raw, str) else raw
                seeds.append(Conversation.from_messages(msgs))
            except Exception:
                continue
    return seeds


@dataclass
class _Item:
    """One candidate row in the shuffled pool, with the fields used for filtering."""

    payload: dict
    success: bool
    round: int
    is_seed: bool


@dataclass
class ViewSampler:
    """Shared sampler backing ``view_past_attempts`` for a whole rotation."""

    store: JsonlStore
    seeds: list[Conversation] = field(default_factory=list)
    true_class_label: str = ""
    target_wrong_label: str = ""  # the class the probe wrongly predicts on a success
    reshuffle: bool = True  # False → show most-recent attempts, seeds as fallback only
    reshuffle_interval: int = 20  # redraw every N submissions (store rows)
    balance: bool = True
    blend_seeds: bool = True
    rng_seed: int = 42
    _interval_idx: int = field(default=-1, init=False)
    _pool: list[_Item] = field(default_factory=list, init=False)

    # ---- rendering -------------------------------------------------------- #

    @staticmethod
    def _render_record(rec: AttemptRecord) -> dict:
        return {
            "messages": [m.to_dict() for m in rec.sample.messages],
            "probe_score": rec.probe_score,
            "probe_predicts": rec.probe_label,
            "judge_label": rec.judge_label or None,
            "judge_reason": rec.judge_reason,
            "success": rec.success,
            "attacker_model": rec.attacker_model,
            "round": rec.round,
        }

    def _seed_items(self) -> list[_Item]:
        reason = (
            "Reference example drawn from the training set (true class), provided as "
            "a seed of what a successful find looks like (not a real submission)."
        )
        items: list[_Item] = []
        for conv in self.seeds:
            payload = {
                "messages": [m.to_dict() for m in conv.messages],
                "probe_score": None,
                "probe_predicts": self.target_wrong_label,
                "judge_label": self.true_class_label,
                "judge_reason": reason,
                "success": True,
                "attacker_model": SEED_ATTACKER_TAG,
                "round": -1,
            }
            items.append(_Item(payload=payload, success=True, round=-1, is_seed=True))
        return items

    def _record_items(self) -> list[_Item]:
        """All real (non-seed) attempts, in insertion (chronological) order."""
        return [
            _Item(self._render_record(rec), rec.success, rec.round, False)
            for rec in self.store.iter_all()
        ]

    @staticmethod
    def _tail(items: list[_Item], k: int) -> list[_Item]:
        """Most-recent ``k`` items (``items[-k:]``), guarding the ``k == 0`` case."""
        return items[-k:] if k > 0 else []

    # ---- reshuffle + sample ---------------------------------------------- #

    def _maybe_reshuffle(self) -> None:
        interval = max(1, self.reshuffle_interval)
        cur = self.store.total_count // interval
        if cur == self._interval_idx and self._pool:
            return
        pool = self._record_items()
        if self.blend_seeds:
            pool.extend(self._seed_items())
        # Seeded RNG keyed on the interval index → reproducible, drift-independent,
        # and a different (but deterministic) shuffle each interval.
        random.Random(f"{self.rng_seed}:{cur}").shuffle(pool)
        self._pool = pool
        self._interval_idx = cur

    def sample(
        self,
        only_successful: bool,
        limit: int,
        current_round: int = 0,
        persistence_from_last_rounds: int | None = None,
    ) -> list[dict]:
        """Return the rendered attempt dicts to show the attacker."""
        if self.reshuffle:
            self._maybe_reshuffle()
            pool = self._pool
        else:
            pool = self._record_items()

        if persistence_from_last_rounds is not None:
            min_round = current_round - persistence_from_last_rounds + 1
            pool = [it for it in pool if it.is_seed or it.round >= min_round]

        succ = [it for it in pool if it.success]
        unsucc = [it for it in pool if not it.success]

        # No-reshuffle mode: training seeds are a fallback for the successful half,
        # used only when there are no real successful attempts yet. (In reshuffle
        # mode seeds are already blended into `succ` by `_maybe_reshuffle`.)
        if not self.reshuffle and not succ and self.blend_seeds:
            succ = self._seed_items()

        # `take` selects the most-recent items when not reshuffling, and the head of
        # the (already-shuffled) pool when reshuffling — equivalent to a random draw.
        take = (lambda items, k: self._tail(items, k)) if not self.reshuffle else (
            lambda items, k: items[:k] if k > 0 else []
        )

        if only_successful:
            chosen = succ if limit <= 0 else take(succ, limit)
        elif not self.balance or limit <= 0:
            chosen = pool if limit <= 0 else take(pool, limit)
        else:
            n_succ = limit - limit // 2  # ceil → the extra one (odd limit) goes to successes
            n_unsucc = limit // 2
            s = take(succ, n_succ)
            u = take(unsucc, n_unsucc)
            # Backfill from the other side so we still reach `limit` when one is short.
            if len(s) < n_succ:
                u = take(unsucc, n_unsucc + (n_succ - len(s)))
            if len(u) < n_unsucc:
                s = take(succ, n_succ + (n_unsucc - len(u)))
            chosen = (s + u)[:limit]

        return [it.payload for it in chosen]
