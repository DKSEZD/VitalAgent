"""
Intervention decision logic for the Proactive Engine.

``InterventionDecision`` is the high-level decision coordinator. It runs
two judge layers against the same tracked health state:

* ``RuleBasedJudge`` for deterministic, guideline-grounded rules.
* ``agent.proactive.llm_judge.ProactiveLLMJudge`` for contextual LLM judgement.

The coordinator merges both judge outputs into a final
``InterventionResult`` while preserving per-layer details for evaluation.

Design goals:

1. **Every rule is grounded in a specific clinical guideline.** Each
   ``ClinicalRule`` carries a ``GuidelineCitation`` that names the
   guideline, section ID, recommendation class where applicable, and a
   short paraphrase. The paper's methodology table can be generated
   directly by iterating ``RuleBasedJudge.rules``.

2. **Rules are data, not code shape.** Adding a rule means appending one
   ``ClinicalRule`` instance to the default registry; the dispatch loop
   is unchanged.

3. **Signal-agnostic core.** The registry is split into three layers:
   ``CORE_RULES`` applies to any cardiac signal (ECG or PPG) and fires
   on generic cardiac concepts (heart rate, rhythm class, episode
   duration); ``ECG_EXTRA_RULES`` adds rules that require ECG-only
   features (morphology, QRS width, ST segment); ``PPG_EXTRA_RULES``
   adds PPG-specific rules (signal quality index, motion artefact
   rejection, etc.). ``RuleBasedJudge`` combines CORE + the extras
   matching the input modality, so the same framework serves both
   signals without duplicating rules that apply to both.

4. **Phase 1 vs Phase 2.** Phase 1 ships three rules in CORE that fire
   on tracker fields already populated today: extreme bradycardia,
   extreme tachycardia, sustained tachycardia at rest. One further
   guideline-grounded rule (long AF episodes) lives in ``PHASE2_RULES``
   and is staged for activation once the tracker begins populating
   rhythm-classification fields. ECG_EXTRA and PPG_EXTRA are empty in
   Phase 1 — placeholders for future morphology / signal-quality rules.

5. **Layering is explicit.** ``RuleBasedJudge.evaluate`` returns a rule
   layer ``InterventionResult``. ``InterventionDecision.evaluate`` returns
   an ``InterventionDecisionOutcome`` containing rule, LLM, and final
   merged results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Force the LLM judge to re-evaluate even when the rule outcome has not
# changed, after this many windows. At the default 10-second window
# length this is roughly 3.3 minutes: short enough that a stable reading
# cannot silently persist for many minutes without LLM re-evaluation,
# long enough to amortise LLM cost across repeated windows.
_JUDGE_FORCE_REFRESH_WINDOWS = 20


# ---------------------------------------------------------------------------
# Modality
# ---------------------------------------------------------------------------


class Modality(str, Enum):
    """
    Signal source this pipeline is processing.

    Future expansion: if a multi-signal fused agent is added, introduce a
    new enum value rather than using a free-form string — callers should
    always be able to switch on a fixed set.
    """

    ECG = "ecg"
    PPG = "ppg"


# ---------------------------------------------------------------------------
# Public enums and result type (unchanged contract with ProactivePipeline)
# ---------------------------------------------------------------------------


class Urgency(str, Enum):
    """Urgency levels for proactive interventions."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class InterventionResult:
    """Intervention output from one judge layer or the merged decision."""

    intervene: bool = False
    urgency: Urgency = Urgency.NONE
    reason: str = ""
    triggered_rules: list[str] | None = None


@dataclass
class InterventionDecisionOutcome:
    """Full high-level decision result with per-judge details."""

    final_result: InterventionResult
    rule_result: InterventionResult
    judge_decision: Any
    judge_cached: bool = False


# ---------------------------------------------------------------------------
# Guideline citation + rule schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuidelineCitation:
    """A pointer into a guideline section that justifies a rule."""

    guideline_id: str
    section_id: str
    cor: str | None
    paraphrase: str
    url: str

    @property
    def fully_qualified_id(self) -> str:
        return f"{self.guideline_id}.{self.section_id}"


RuleTrigger = Callable[[dict[str, Any]], str | None]


@dataclass(frozen=True)
class ClinicalRule:
    """
    A single deterministic clinical rule.

    Fields:
        rule_id: short stable identifier, surfaced in ``triggered_rules``
        description: one-line human description
        urgency: fixed urgency tier this rule emits when it fires
        citation: the guideline recommendation this rule is derived from
        trigger: function that inspects health state and returns the
                 reason string (or None for no-fire)
    """

    rule_id: str
    description: str
    urgency: Urgency
    citation: GuidelineCitation
    trigger: RuleTrigger


# ---------------------------------------------------------------------------
# Triggers for the Phase-1 CORE rule set
# ---------------------------------------------------------------------------
#
# CORE triggers depend only on signal-agnostic cardiac fields:
#   short.hr_bpm                   — beats per minute (current window)
#   short.rhythm_class             — "N" | "AF" | "AFL" | "Other" | None
#   short.af_episode_duration_s    — seconds the current AF episode has lasted
#   short.tachycardia_ratio_5min   — fraction of the trailing 5-minute
#                                    window with HR > 100 bpm
#   short.tachycardia_sample_count — number of HR samples in that trailing
#                                    window (used to require a full 5-min
#                                    observation before the rule can fire)
#
# These fields can be populated from either ECG or PPG; the rule is
# indifferent to how they were computed upstream.


_HR_EXTREME_LOW_BPM = 30.0
_HR_EXTREME_LOW_MIN_CONSECUTIVE_S = 10.0
_HR_EXTREME_HIGH_BPM = 200.0
_HR_EXTREME_HIGH_MIN_CONSECUTIVE_S = 10.0
_AF_LONG_EPISODE_S = 24 * 3600  # 24 hours

# Sustained tachycardia: 5-minute rolling window, >= 80% of windows with
# HR > 100 bpm, and at least a full window worth of samples collected.
# 100 bpm threshold per SVT-2015 sinus-tachycardia definition; the 80% /
# 5-minute gate is a conservative implementation choice to avoid firing
# on routine activity (brisk walking, stair climbing), not a guideline
# threshold.
_TACHY_SUSTAINED_RATIO = 0.80
_TACHY_MIN_SAMPLES = 30  # at 10-s windows = 5 minutes of observation


def _short(state: dict[str, Any]) -> dict[str, Any]:
    return state.get("short") or {}


def _hr_sustained_seconds(state: dict[str, Any]) -> float:
    """
    Duration over which the current HR reading has been observed.

    Placeholder: today's tracker does not expose a 'HR stable for N
    seconds' field. For the Phase-1 rewrite we assume a window-level
    reading represents the full window duration and rely on upstream
    window length (typically 10 s). Day 5 replaces this with a
    tracker-owned sustain counter.
    """
    meta = state.get("window_meta") or {}
    return float(meta.get("window_duration_s", 0.0))


def _trigger_extreme_bradycardia(state: dict[str, Any]) -> str | None:
    hr = _short(state).get("hr_bpm")
    if hr is None:
        return None
    if hr >= _HR_EXTREME_LOW_BPM:
        return None
    if _hr_sustained_seconds(state) < _HR_EXTREME_LOW_MIN_CONSECUTIVE_S:
        return None
    return (
        f"Heart rate {hr:.0f} bpm is below the critical-bradycardia "
        f"threshold of {_HR_EXTREME_LOW_BPM:.0f} bpm."
    )


def _trigger_extreme_tachycardia(state: dict[str, Any]) -> str | None:
    hr = _short(state).get("hr_bpm")
    if hr is None:
        return None
    if hr <= _HR_EXTREME_HIGH_BPM:
        return None
    if _hr_sustained_seconds(state) < _HR_EXTREME_HIGH_MIN_CONSECUTIVE_S:
        return None
    return (
        f"Heart rate {hr:.0f} bpm is above the critical-tachycardia "
        f"threshold of {_HR_EXTREME_HIGH_BPM:.0f} bpm."
    )


def _trigger_long_af_episode(state: dict[str, Any]) -> str | None:
    short = _short(state)
    rhythm = short.get("rhythm_class")
    duration = short.get("af_episode_duration_s")
    if rhythm != "AF":
        return None
    if duration is None:
        return None
    if duration < _AF_LONG_EPISODE_S:
        return None
    hours = duration / 3600.0
    return (
        f"Atrial fibrillation episode has been sustained for {hours:.1f} hours, "
        "exceeding the 24-hour threshold in the AF guideline's AHRE framework."
    )


def _trigger_sustained_tachycardia(state: dict[str, Any]) -> str | None:
    """
    Fire when resting heart rate has been elevated for sustained periods.

    The SVT guideline defines sinus tachycardia as sinus rate exceeding
    100 bpm, and identifies resting HR > 100 bpm as the defining feature
    of inappropriate sinus tachycardia. Because we cannot detect
    "at rest" directly without an activity channel, we use sustained
    duration as a conservative proxy: firing only when at least
    ``_TACHY_SUSTAINED_RATIO`` of the last ``_TACHY_MIN_SAMPLES``
    readings have been above 100 bpm. At 10-second windows, this is a
    full 5 minutes of predominantly elevated heart rate — unlikely to
    reflect routine brief exertion.
    """
    short = _short(state)
    ratio = short.get("tachycardia_ratio_5min")
    samples = short.get("tachycardia_sample_count", 0)

    if ratio is None:
        return None
    if samples < _TACHY_MIN_SAMPLES:
        return None
    if ratio < _TACHY_SUSTAINED_RATIO:
        return None

    pct = ratio * 100
    return (
        f"Heart rate has exceeded 100 bpm in {pct:.0f}% of the last "
        f"{samples} readings (approximately 5 minutes), consistent with "
        "sustained tachycardia as defined in the SVT guideline."
    )


# ---------------------------------------------------------------------------
# Rule registries
# ---------------------------------------------------------------------------
# CORE: rules that apply to any cardiac monitoring modality.
# ECG_EXTRA: rules that require ECG-only features (morphology, QRS, ST).
# PPG_EXTRA: rules that require PPG-only features (optical SQI, motion).
# PHASE2: rules whose upstream tracker fields are not yet populated by
#         the current tracker implementation. These rules are
#         guideline-grounded and ready to fire, but do not appear in the
#         default active rule set until a downstream component
#         (e.g. the rhythm classifier planned for Day 5) starts
#         providing the inputs they read. Explicitly opting in via
#         ``include_phase2=True`` activates them — useful for tests and
#         for when the tracker has caught up.
#
# Phase 1 active rules (tracker-backed today): CORE has three rules,
# both EXTRA registries are empty.
# Phase 2 staged rules (tracker work pending): PHASE2 has one rule
# (long_af_episode). Migrates into CORE once inputs are available.


CORE_RULES: tuple[ClinicalRule, ...] = (
    ClinicalRule(
        rule_id="extreme_bradycardia",
        description=(
            "Sustained heart rate below 30 bpm (pre-arrest range, awake or "
            "unknown state)"
        ),
        urgency=Urgency.CRITICAL,
        citation=GuidelineCitation(
            guideline_id="brady-2018",
            section_id="av-block-ventricular-rate",
            cor="Class 1",
            paraphrase=(
                "High-grade AV block with an awake ventricular escape rate "
                "below 40 bpm is a Class 1 pacing indication; sustained "
                "rates below 30 bpm fall within that actionable range and "
                "warrant immediate clinical evaluation."
            ),
            url=(
                "https://www.ahajournals.org/doi/10.1161/"
                "CIR.0000000000000628"
            ),
        ),
        trigger=_trigger_extreme_bradycardia,
    ),
    ClinicalRule(
        rule_id="extreme_tachycardia",
        description=(
            "Sustained heart rate above 200 bpm (within acute-management "
            "range for SVT / VT)"
        ),
        urgency=Urgency.CRITICAL,
        citation=GuidelineCitation(
            guideline_id="svt-2015",
            section_id="avnrt-rate",
            cor=None,
            paraphrase=(
                "AVNRT, the most common SVT, typically presents at "
                "180-200 bpm and can exceed 250 bpm; sustained "
                "ventricular rates above 200 bpm are within the range "
                "that warrants acute clinical management."
            ),
            url=(
                "https://www.ahajournals.org/doi/10.1161/"
                "CIR.0000000000000311"
            ),
        ),
        trigger=_trigger_extreme_tachycardia,
    ),
    ClinicalRule(
        rule_id="sustained_tachycardia",
        description=(
            "Heart rate above 100 bpm sustained for a majority of a "
            "5-minute window"
        ),
        urgency=Urgency.MEDIUM,
        citation=GuidelineCitation(
            guideline_id="svt-2015",
            section_id="tachycardia-definition",
            cor=None,
            paraphrase=(
                "The SVT guideline defines tachycardia as heart rate above "
                "100 bpm and identifies resting HR > 100 bpm as the "
                "defining feature of inappropriate sinus tachycardia. "
                "Sustained elevation over a 5-minute window, unless "
                "exertion-explained, warrants clinical evaluation to "
                "exclude reversible causes."
            ),
            url=(
                "https://www.ahajournals.org/doi/10.1161/"
                "CIR.0000000000000311"
            ),
        ),
        trigger=_trigger_sustained_tachycardia,
    ),
)


ECG_EXTRA_RULES: tuple[ClinicalRule, ...] = ()
"""ECG-only rules. Populated in Phase 2 once morphology features
(QRS width, ST elevation, T-wave inversion, etc.) are exposed by the
tracker."""


PPG_EXTRA_RULES: tuple[ClinicalRule, ...] = ()
"""PPG-only rules. Populated once PPG-specific features are available
(signal quality index, motion-artefact flag, perfusion index, etc.)."""


# Phase 2 staged rules — guideline-grounded but waiting on tracker inputs.
# Do not remove citations or trigger functions when moving a rule in or
# out of this tuple; they are the paper's methodology record.
PHASE2_RULES: tuple[ClinicalRule, ...] = (
    ClinicalRule(
        rule_id="long_af_episode",
        description=(
            "Atrial fibrillation episode sustained for 24 hours or more"
        ),
        urgency=Urgency.HIGH,
        citation=GuidelineCitation(
            guideline_id="af-2023",
            section_id="ahre-24h-plus",
            cor="Class 2a",
            paraphrase=(
                "For device-detected AHRE lasting 24 hours or longer and "
                "CHA2DS2-VASc >= 2, the guideline states it is reasonable "
                "to initiate anticoagulation within shared decision-making. "
                "Because stroke-risk context is not available in-device, "
                "any such episode is routed to clinical evaluation."
            ),
            url=(
                "https://www.ahajournals.org/doi/10.1161/"
                "CIR.0000000000001193"
            ),
        ),
        trigger=_trigger_long_af_episode,
    ),
)
"""Rules whose tracker-side inputs are not populated yet.

At the time of writing, the tracker exposes ``short.rhythm_regular``
(boolean) but not ``short.rhythm_class`` or
``short.af_episode_duration_s``. Until the rhythm classifier and AF
episode tracker land (planned for Day 5), rules that depend on those
fields cannot fire via the real pipeline. They are kept in this
separate registry so the rule-dispatch code itself doesn't need to
branch on "is this rule's input available", and unit tests can pass
them to ``RuleBasedJudge(rules=...)`` directly to verify the
trigger logic.

Activation path: once ``HealthStateTracker`` starts populating the
missing fields, either migrate the rule back into ``CORE_RULES`` or
construct the decision layer with ``include_phase2=True``."""


def rules_for_modality(
    modality: Modality,
    include_phase2: bool = False,
) -> tuple[ClinicalRule, ...]:
    """
    Return the active rule set for a given signal modality.

    Parameters
    ----------
    modality:
        ECG or PPG. Determines which EXTRA registry is unioned with CORE.
    include_phase2:
        When True, also include ``PHASE2_RULES``. Defaults to False so
        production pipelines only run rules whose inputs the tracker
        actually populates. Flip to True once tracker-side prerequisites
        are in place, or pass True from tests exercising staged rules.
    """
    if modality == Modality.ECG:
        base = CORE_RULES + ECG_EXTRA_RULES
    elif modality == Modality.PPG:
        base = CORE_RULES + PPG_EXTRA_RULES
    else:
        raise ValueError(f"Unknown modality: {modality!r}")

    if include_phase2:
        return base + PHASE2_RULES
    return base


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------


class RuleBasedJudge:
    """
    Guideline-grounded rule judge.

    Given a signal modality, evaluates the current health state against
    the active rule set (CORE + modality-specific extras) and returns an
    ``InterventionResult`` whose urgency is the maximum over all fired
    rules.
    """

    def __init__(
        self,
        modality: Modality = Modality.ECG,
        rules: tuple[ClinicalRule, ...] | None = None,
        include_phase2: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        modality:
            ECG or PPG. Drives which EXTRA registry is unioned with CORE.
        rules:
            Override the active rule set entirely. When provided,
            ``modality`` and ``include_phase2`` are ignored for the
            purpose of rule selection (modality is still retained for
            informational output).
        include_phase2:
            When True, also activate ``PHASE2_RULES``. Default False —
            only rules whose tracker inputs are currently populated
            will run. See ``PHASE2_RULES`` docstring for what is
            currently staged.
        """
        self.modality = modality
        self.include_phase2 = include_phase2
        self.rules: tuple[ClinicalRule, ...] = (
            rules
            if rules is not None
            else rules_for_modality(modality, include_phase2=include_phase2)
        )
        logger.info(
            "RuleBasedJudge initialised: modality=%s, phase2=%s, "
            "%d rule(s): %s",
            self.modality.value,
            self.include_phase2,
            len(self.rules),
            [r.rule_id for r in self.rules],
        )

    def evaluate(
        self,
        health_state: dict[str, Any],
        embedding_zscore: float = 0.0,
    ) -> InterventionResult:
        """
        Run every registered rule's trigger against the health state.

        Parameters
        ----------
        health_state : dict
            Output of ``HealthStateTracker.get_current_state()``.
        embedding_zscore : float
            Present for interface compatibility. The embedding channel is
            stubbed in Phase 1 and this value is ignored by guideline-
            grounded rules.
        """
        del embedding_zscore  # unused; kept for pipeline compatibility

        triggered_rule_ids: list[str] = []
        reasons: list[str] = []
        highest = Urgency.NONE

        for rule in self.rules:
            try:
                reason = rule.trigger(health_state)
            except Exception:
                logger.exception("Rule trigger raised: %s", rule.rule_id)
                continue
            if reason is None:
                continue
            triggered_rule_ids.append(rule.rule_id)
            reasons.append(reason)
            if _urgency_rank(rule.urgency) > _urgency_rank(highest):
                highest = rule.urgency

        if not triggered_rule_ids:
            return InterventionResult()

        return InterventionResult(
            intervene=True,
            urgency=highest,
            reason=" ".join(reasons),
            triggered_rules=triggered_rule_ids,
        )

    def get_rule(self, rule_id: str) -> ClinicalRule | None:
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    def describe_rules(self) -> list[dict[str, Any]]:
        """Serialisable description of every active rule."""
        phase2_ids = {r.rule_id for r in PHASE2_RULES}
        return [
            {
                "rule_id": r.rule_id,
                "description": r.description,
                "urgency": r.urgency.value,
                "modality": self.modality.value,
                "phase": "phase-2-staged" if r.rule_id in phase2_ids else "phase-1-active",
                "guideline_id": r.citation.guideline_id,
                "section_id": r.citation.section_id,
                "cor": r.citation.cor,
                "paraphrase": r.citation.paraphrase,
                "url": r.citation.url,
            }
            for r in self.rules
        ]

    @staticmethod
    def describe_all_rules(modality: Modality = Modality.ECG) -> list[dict[str, Any]]:
        """
        Describe every rule in the catalogue (Phase 1 + Phase 2) for
        the given modality, regardless of current activation state.

        Useful for paper methodology tables that want to show both the
        active rules and the staged-for-next-phase rules in one place.
        """
        all_rules = rules_for_modality(modality, include_phase2=True)
        phase2_ids = {r.rule_id for r in PHASE2_RULES}
        return [
            {
                "rule_id": r.rule_id,
                "description": r.description,
                "urgency": r.urgency.value,
                "modality": modality.value,
                "phase": "phase-2-staged" if r.rule_id in phase2_ids else "phase-1-active",
                "guideline_id": r.citation.guideline_id,
                "section_id": r.citation.section_id,
                "cor": r.citation.cor,
                "paraphrase": r.citation.paraphrase,
                "url": r.citation.url,
            }
            for r in all_rules
        ]


class InterventionDecision:
    """
    High-level proactive intervention decision coordinator.

    Runs the deterministic rule-based judge and the LLM judge on the
    same health state, then merges their outputs into the final
    intervention result. Per-layer outputs are returned for evaluation,
    debugging, and ablation reports.
    """

    def __init__(
        self,
        modality: Modality = Modality.ECG,
        rule_judge: RuleBasedJudge | None = None,
        llm_judge: Any = None,
        rules: tuple[ClinicalRule, ...] | None = None,
        include_phase2: bool = False,
    ) -> None:
        self.modality = modality
        self.rule_judge = (
            rule_judge
            if rule_judge is not None
            else RuleBasedJudge(
                modality=modality,
                rules=rules,
                include_phase2=include_phase2,
            )
        )
        self.llm_judge = llm_judge if llm_judge is not None else self._default_llm_judge()

        self._judge_cache: Any = None
        self._last_judge_signature: tuple | None = None
        self._windows_since_judge_call: int = 0

        logger.info(
            "InterventionDecision initialised: modality=%s, rules=%d, llm_judge=%s.",
            self.modality.value,
            len(self.rules),
            "active" if self.llm_judge.enabled else "stub",
        )

    @property
    def rules(self) -> tuple[ClinicalRule, ...]:
        return self.rule_judge.rules

    def evaluate(
        self,
        health_state: dict[str, Any],
        embedding_zscore: float = 0.0,
    ) -> InterventionDecisionOutcome:
        rule_result = self.rule_judge.evaluate(
            health_state,
            embedding_zscore=embedding_zscore,
        )
        judge_decision, judge_cached = self._maybe_invoke_judge(health_state, rule_result)
        judge_result = judge_decision.to_intervention_result()
        final_result = _merge_results(rule_result, judge_result)
        return InterventionDecisionOutcome(
            final_result=final_result,
            rule_result=rule_result,
            judge_decision=judge_decision,
            judge_cached=judge_cached,
        )

    def get_rule(self, rule_id: str) -> ClinicalRule | None:
        return self.rule_judge.get_rule(rule_id)

    def describe_rules(self) -> list[dict[str, Any]]:
        return self.rule_judge.describe_rules()

    @staticmethod
    def describe_all_rules(modality: Modality = Modality.ECG) -> list[dict[str, Any]]:
        return RuleBasedJudge.describe_all_rules(modality=modality)

    def _default_llm_judge(self) -> Any:
        from agent.proactive.llm_judge import ProactiveLLMJudge

        return ProactiveLLMJudge(modality=self.modality)

    def _maybe_invoke_judge(
        self,
        state: dict[str, Any],
        rule_result: InterventionResult,
    ) -> tuple[Any, bool]:
        """
        Invoke the LLM judge, reusing the previous decision if the rule
        layer's outcome has not changed.
        """
        if not self.llm_judge.enabled:
            # Stub mode is effectively free (coarse gating plus no-alert),
            # so caching would only obscure what happened in each window.
            return self.llm_judge.evaluate(state), False

        sig = _rule_outcome_signature(rule_result)
        force_refresh = (
            self._last_judge_signature is not None
            and self._windows_since_judge_call >= _JUDGE_FORCE_REFRESH_WINDOWS
        )
        cache_hit = (
            sig == self._last_judge_signature
            and self._judge_cache is not None
            and not force_refresh
        )
        if cache_hit:
            self._windows_since_judge_call += 1
            logger.debug(
                "Judge cache hit (signature=%s, windows_since=%d)",
                sig,
                self._windows_since_judge_call,
            )
            return self._judge_cache, True

        if force_refresh:
            logger.debug(
                "Judge cache force-refresh after %d windows (signature=%s)",
                self._windows_since_judge_call,
                sig,
            )

        decision = self.llm_judge.evaluate(state)
        self._judge_cache = decision
        self._last_judge_signature = sig
        self._windows_since_judge_call = 0
        return decision, False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_results(
    rule_result: InterventionResult,
    judge_result: InterventionResult,
) -> InterventionResult:
    """Combine rule-based and LLM-judge outputs into one result."""
    rule_rank = _urgency_rank(rule_result.urgency)
    judge_rank = _urgency_rank(judge_result.urgency)

    if not rule_result.intervene and not judge_result.intervene:
        return InterventionResult()

    merged_urgency = (
        rule_result.urgency if rule_rank >= judge_rank else judge_result.urgency
    )

    triggered: list[str] = []
    for source in (rule_result.triggered_rules or [], judge_result.triggered_rules or []):
        for name in source:
            if name not in triggered:
                triggered.append(name)

    reason_parts = [r for r in (rule_result.reason, judge_result.reason) if r]
    return InterventionResult(
        intervene=True,
        urgency=merged_urgency,
        reason=" ".join(reason_parts),
        triggered_rules=triggered,
    )


def _rule_outcome_signature(rule_result: InterventionResult) -> tuple:
    return (
        frozenset(rule_result.triggered_rules or []),
        rule_result.urgency,
    )


def _urgency_rank(u: Urgency) -> int:
    return {
        Urgency.NONE: 0,
        Urgency.LOW: 1,
        Urgency.MEDIUM: 2,
        Urgency.HIGH: 3,
        Urgency.CRITICAL: 4,
    }.get(u, 0)
