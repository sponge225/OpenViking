def phase1_fallback_judgment(
    record: dict,
    trigger_field: str | None = None,
    provider_name: str | None = None,
) -> dict:
    fallback = (record or {}).get("fallback", {}) or {}
    metrics = (record or {}).get("metrics", {}) or {}

    provider_result = {}
    if provider_name:
        provider_result = (fallback.get("provider_results", {}) or {}).get(provider_name, {}) or {}
        judged_trigger = bool(provider_result.get("should_fallback"))
    elif trigger_field:
        judged_trigger = bool(fallback.get(trigger_field))
    elif "phase1_judge_should_fallback" in fallback:
        judged_trigger = bool(fallback.get("phase1_judge_should_fallback"))
    else:
        judged_trigger = bool(fallback.get("triggered"))

    provider_metrics = provider_result.get("metrics", {}) if provider_result else {}
    if provider_metrics and "Accuracy" in provider_metrics:
        accuracy_source = f"Provider Accuracy: {provider_name}"
        accuracy_value = provider_metrics.get("Accuracy")
    else:
        accuracy_source = "Phase1 Accuracy" if "Phase1 Accuracy" in metrics else "Accuracy"
        accuracy_value = metrics.get(accuracy_source, 0)
    try:
        accuracy = float(accuracy_value or 0)
    except (TypeError, ValueError):
        accuracy = 0.0

    should_trigger = accuracy <= 2.0
    should_not_trigger = accuracy >= 3.0
    bucketed = should_trigger or should_not_trigger
    false_positive = bucketed and judged_trigger and should_not_trigger
    false_negative = bucketed and (not judged_trigger) and should_trigger

    if false_positive:
        error_type = "false_positive"
    elif false_negative:
        error_type = "false_negative"
    elif not bucketed:
        error_type = "unbucketed_accuracy"
    elif judged_trigger:
        error_type = "correct_trigger"
    else:
        error_type = "correct_not_trigger"

    return {
        "judged_trigger": judged_trigger,
        "expected_trigger": should_trigger if bucketed else None,
        "accuracy": accuracy,
        "accuracy_source": accuracy_source,
        "bucketed": bucketed,
        "error": false_positive or false_negative,
        "error_type": error_type,
    }


def phase1_fallback_judgment_report(
    records: list[dict],
    trigger_field: str | None = None,
    provider_name: str | None = None,
) -> dict:
    judgments = [
        phase1_fallback_judgment(
            r,
            trigger_field=trigger_field,
            provider_name=provider_name,
        )
        for r in records
    ]
    bucketed = [j for j in judgments if j["bucketed"]]
    false_positive = [j for j in bucketed if j["error_type"] == "false_positive"]
    false_negative = [j for j in bucketed if j["error_type"] == "false_negative"]
    judged_trigger = [j for j in bucketed if j["judged_trigger"]]
    judged_not_trigger = [j for j in bucketed if not j["judged_trigger"]]
    expected_trigger = [j for j in bucketed if j["expected_trigger"]]
    expected_not_trigger = [j for j in bucketed if not j["expected_trigger"]]
    phase1_accuracy_records = [j for j in bucketed if j.get("accuracy_source") == "Phase1 Accuracy"]
    provider_accuracy_records = [
        j for j in bucketed
        if str(j.get("accuracy_source", "")).startswith("Provider Accuracy:")
    ]
    total = len(bucketed)
    avg_accuracy = sum(j["accuracy"] for j in bucketed) / total if total else 0.0
    avg_triggered_accuracy = (
        sum(j["accuracy"] for j in judged_trigger) / len(judged_trigger)
        if judged_trigger else 0.0
    )
    avg_not_triggered_accuracy = (
        sum(j["accuracy"] for j in judged_not_trigger) / len(judged_not_trigger)
        if judged_not_trigger else 0.0
    )

    return {
        "Total Records": len(records),
        "Judgment Accuracy Source": (
            f"Provider Accuracy: {provider_name}" if provider_name and len(provider_accuracy_records) == total and total else
            "Phase1 Accuracy" if len(phase1_accuracy_records) == total and total else
            "Mixed Provider/Phase1 Accuracy" if provider_accuracy_records else
            "Mixed Phase1 Accuracy/Accuracy" if phase1_accuracy_records else
            "Accuracy"
        ),
        "Bucketed Accuracy Records": total,
        "Unbucketed Accuracy Count": len(judgments) - total,
        "Judged Trigger Count": len(judged_trigger),
        "Judged Not Trigger Count": len(judged_not_trigger),
        "Average Accuracy (judged triggered)": avg_triggered_accuracy,
        "Average Accuracy (judged not triggered)": avg_not_triggered_accuracy,
        "Average Accuracy (overall)": avg_accuracy,
        "Expected Trigger Count (Accuracy 0-2)": len(expected_trigger),
        "Expected Not Trigger Count (Accuracy 3-4)": len(expected_not_trigger),
        "False Positive Count (triggered but Accuracy 3-4)": len(false_positive),
        "False Negative Count (not triggered but Accuracy 0-2)": len(false_negative),
        "Judgment Error Count": len(false_positive) + len(false_negative),
        "Judgment Error Rate": (
            (len(false_positive) + len(false_negative)) / total
            if total else 0.0
        ),
        "False Positive Rate Among Judged Triggered": (
            len(false_positive) / len(judged_trigger)
            if judged_trigger else 0.0
        ),
        "False Negative Rate Among Judged Not Triggered": (
            len(false_negative) / len(judged_not_trigger)
            if judged_not_trigger else 0.0
        ),
        "Recall for Bad Phase1 Answers": (
            (len(expected_trigger) - len(false_negative)) / len(expected_trigger)
            if expected_trigger else 0.0
        ),
        "Precision for Triggered Fallback": (
            (len(judged_trigger) - len(false_positive)) / len(judged_trigger)
            if judged_trigger else 0.0
        ),
    }


def fallback_judgment_summary(records: list[dict]) -> dict:
    report = phase1_fallback_judgment_report(records)
    total = int(report.get("Bucketed Accuracy Records", 0) or 0)
    error_count = int(report.get("Judgment Error Count", 0) or 0)
    error_rate = float(report.get("Judgment Error Rate", 0.0) or 0.0)
    return {
        "Total Records": int(report.get("Total Records", 0) or 0),
        "Total Judged (records with Phase1 Accuracy 0-2 or 3-4)": total,
        "Unbucketed Accuracy Count (Phase1 Accuracy not in 0-2 or 3-4)": int(
            report.get("Unbucketed Accuracy Count", 0) or 0
        ),
        "Triggered Count": int(report.get("Judged Trigger Count", 0) or 0),
        "Not Triggered Count": int(report.get("Judged Not Trigger Count", 0) or 0),
        "Expected Trigger Count (Phase1 Accuracy 0-2)": int(
            report.get("Expected Trigger Count (Accuracy 0-2)", 0) or 0
        ),
        "Expected Not Trigger Count (Phase1 Accuracy 3-4)": int(
            report.get("Expected Not Trigger Count (Accuracy 3-4)", 0) or 0
        ),
        "False Positive Count (triggered but Phase1 Accuracy 3-4)": int(
            report.get("False Positive Count (triggered but Accuracy 3-4)", 0) or 0
        ),
        "False Negative Count (not triggered but Phase1 Accuracy 0-2)": int(
            report.get("False Negative Count (not triggered but Accuracy 0-2)", 0) or 0
        ),
        "Judgment Error Count": error_count,
        "Judgment Error Rate": error_rate,
        "Judgment Accuracy (1 - Judgment Error Rate)": (
            1.0 - error_rate if total else 0.0
        ),
    }


def fallback_miss_summary(records: list[dict]) -> dict:
    report = phase1_fallback_judgment_report(records)
    total_records = int(report.get("Total Records", 0) or 0)
    not_triggered = int(report.get("Judged Not Trigger Count", 0) or 0)
    expected_trigger = int(report.get("Expected Trigger Count (Accuracy 0-2)", 0) or 0)
    miss_count = int(report.get("False Negative Count (not triggered but Accuracy 0-2)", 0) or 0)
    return {
        "Miss Count (not triggered but Phase1 Accuracy 0-2)": miss_count,
        "Miss Rate Overall (Miss Count / Total Records)": (
            miss_count / total_records if total_records else 0.0
        ),
        "Miss Rate Among Not Triggered (Miss Count / Not Triggered Count)": (
            miss_count / not_triggered if not_triggered else 0.0
        ),
        "Recall for Bad Phase1 Answers (triggered bad Phase1 / all bad Phase1)": (
            (expected_trigger - miss_count) / expected_trigger
            if expected_trigger else 0.0
        ),
    }


def recoverable_miss_summary(records: list[dict]) -> dict:
    total_records = len(records)
    not_triggered_records = [
        r for r in records
        if not ((r.get("fallback", {}) or {}).get("triggered"))
    ]
    shadow_evaluated = [
        r for r in not_triggered_records
        if ((r.get("fallback", {}) or {}).get("shadow_bot", {}) or {}).get("executed")
    ]

    fallback_misses = [
        record for record in records
        if phase1_fallback_judgment(record).get("error_type") == "false_negative"
    ]

    recoverable = []
    non_recoverable = []
    unscored = []
    for record in fallback_misses:
        shadow_bot = ((record.get("fallback", {}) or {}).get("shadow_bot", {}) or {})
        shadow_metrics = shadow_bot.get("metrics", {}) or {}
        if "Accuracy" not in shadow_metrics:
            unscored.append(record)
            continue
        try:
            shadow_accuracy = float(shadow_metrics.get("Accuracy", 0.0) or 0.0)
        except (TypeError, ValueError):
            unscored.append(record)
            continue
        if shadow_accuracy >= 3.0:
            recoverable.append(record)
        elif shadow_accuracy <= 2.0:
            non_recoverable.append(record)
        else:
            unscored.append(record)

    miss_count = len(fallback_misses)
    not_triggered_count = len(not_triggered_records)
    return {
        "Shadow Bot Evaluated Count (not-triggered records evaluated by bot for counterfactual only)": len(shadow_evaluated),
        "Fallback Miss Count (not triggered and Phase1 Accuracy 0-2)": miss_count,
        "Fallback Miss Count With Shadow Bot Accuracy": miss_count - len(unscored),
        "Recoverable Miss Count (not triggered, Phase1 Accuracy 0-2, Shadow Bot Accuracy 3-4)": len(recoverable),
        "Non-Recoverable Miss Count (not triggered, Phase1 Accuracy 0-2, Shadow Bot Accuracy 0-2)": len(non_recoverable),
        "Unscored Fallback Miss Count (missing shadow bot accuracy)": len(unscored),
        "Recoverable Miss Rate Overall (Recoverable Miss Count / Total Records)": (
            len(recoverable) / total_records if total_records else 0.0
        ),
        "Recoverable Miss Rate Among Misses (Recoverable Miss Count / Fallback Miss Count)": (
            len(recoverable) / miss_count if miss_count else 0.0
        ),
        "Recoverable Miss Rate Among Not Triggered (Recoverable Miss Count / Not Triggered Count)": (
            len(recoverable) / not_triggered_count if not_triggered_count else 0.0
        ),
    }


def _record_label(record: dict) -> str:
    return (
        f"sample_id={record.get('sample_id', '')}, "
        f"query_id={record.get('_global_index', '')}"
    )


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _phase1_accuracy_for_gain(record: dict, provider_name: str | None = None) -> tuple[float | None, str]:
    fallback = (record or {}).get("fallback", {}) or {}
    provider_results = fallback.get("provider_results", {}) or {}
    selected_provider = provider_name or fallback.get("primary_provider")
    if selected_provider:
        provider_result = provider_results.get(selected_provider, {}) or {}
        provider_metrics = provider_result.get("metrics", {}) or {}
        if "Accuracy" in provider_metrics:
            return _float_or_none(provider_metrics.get("Accuracy")), f"provider:{selected_provider}"

    metrics = (record or {}).get("metrics", {}) or {}
    if "Phase1 Accuracy" in metrics:
        return _float_or_none(metrics.get("Phase1 Accuracy")), "Phase1 Accuracy"
    return None, "missing phase1/provider accuracy"


def _bot_accuracy_for_gain(record: dict) -> tuple[float | None, str]:
    fallback = (record or {}).get("fallback", {}) or {}
    metrics = (record or {}).get("metrics", {}) or {}

    if fallback.get("executed"):
        if "Accuracy" in metrics:
            return _float_or_none(metrics.get("Accuracy")), "final bot Accuracy"
        return None, "fallback executed but final Accuracy is missing"

    shadow_bot = fallback.get("shadow_bot", {}) or {}
    if not shadow_bot:
        return None, "fallback not triggered and shadow_bot record is missing"
    if not shadow_bot.get("executed"):
        error = shadow_bot.get("error", "")
        if error:
            return None, f"shadow bot not executed: {error}"
        return None, "shadow bot not executed"

    shadow_metrics = shadow_bot.get("metrics", {}) or {}
    if "Accuracy" not in shadow_metrics:
        return None, "shadow bot executed but shadow metrics are missing"
    return _float_or_none(shadow_metrics.get("Accuracy")), "shadow bot Accuracy"


def _log_gain_unscored(logger, record: dict, reason: str) -> None:
    if not logger:
        return
    logger.error(
        f"[FallbackGainMetrics][UNSCORED] {_record_label(record)} | {reason}"
    )


def fallback_gain_judgments(
    records: list[dict],
    provider_name: str | None = None,
    margin: float = 0.0,
    logger=None,
) -> list[dict]:
    judgments = []
    for record in records:
        fallback = (record or {}).get("fallback", {}) or {}
        judged_trigger = bool(
            fallback.get(
                "phase1_judge_should_fallback",
                fallback.get("triggered", False),
            )
        )
        phase1_acc, phase1_source = _phase1_accuracy_for_gain(record, provider_name=provider_name)
        bot_acc, bot_source = _bot_accuracy_for_gain(record)
        unscored_reasons = []
        if phase1_acc is None:
            unscored_reasons.append(phase1_source)
        if bot_acc is None:
            unscored_reasons.append(bot_source)

        if unscored_reasons:
            reason = "; ".join(unscored_reasons)
            _log_gain_unscored(logger, record, reason)
            judgments.append({
                "record": record,
                "judged_trigger": judged_trigger,
                "phase1_accuracy": phase1_acc,
                "phase1_accuracy_source": phase1_source,
                "bot_accuracy": bot_acc,
                "bot_accuracy_source": bot_source,
                "accuracy_gain": None,
                "expected_trigger": None,
                "scored": False,
                "unscored_reason": reason,
                "error": False,
                "error_type": "unscored",
            })
            continue

        gain = bot_acc - phase1_acc
        expected_trigger = gain > margin
        false_positive = judged_trigger and not expected_trigger
        false_negative = (not judged_trigger) and expected_trigger
        if false_positive:
            error_type = "false_positive"
        elif false_negative:
            error_type = "false_negative"
        elif judged_trigger:
            error_type = "correct_trigger"
        else:
            error_type = "correct_not_trigger"

        judgments.append({
            "record": record,
            "judged_trigger": judged_trigger,
            "phase1_accuracy": phase1_acc,
            "phase1_accuracy_source": phase1_source,
            "bot_accuracy": bot_acc,
            "bot_accuracy_source": bot_source,
            "accuracy_gain": gain,
            "expected_trigger": expected_trigger,
            "scored": True,
            "unscored_reason": "",
            "error": false_positive or false_negative,
            "error_type": error_type,
        })
    return judgments


def fallback_judgment_gain_summary(
    records: list[dict],
    provider_name: str | None = None,
    margin: float = 0.0,
    logger=None,
) -> dict:
    judgments = fallback_gain_judgments(
        records,
        provider_name=provider_name,
        margin=margin,
        logger=logger,
    )
    scored = [j for j in judgments if j["scored"]]
    unscored = [j for j in judgments if not j["scored"]]
    judged_trigger = [j for j in scored if j["judged_trigger"]]
    judged_not_trigger = [j for j in scored if not j["judged_trigger"]]
    expected_trigger = [j for j in scored if j["expected_trigger"]]
    expected_not_trigger = [j for j in scored if not j["expected_trigger"]]
    false_positive = [j for j in scored if j["error_type"] == "false_positive"]
    false_negative = [j for j in scored if j["error_type"] == "false_negative"]
    total_scored = len(scored)
    error_count = len(false_positive) + len(false_negative)
    unscored_by_reason: dict[str, int] = {}
    for j in unscored:
        reason = j.get("unscored_reason", "unknown")
        unscored_by_reason[reason] = unscored_by_reason.get(reason, 0) + 1

    return {
        "Total Records": len(records),
        "Total Scored Records (records with both Phase1 and Bot Accuracy)": total_scored,
        "Unscored Count (missing Phase1 or Bot Accuracy)": len(unscored),
        "Unscored Reasons": unscored_by_reason,
        "Accuracy Gain Margin": margin,
        "Triggered Count": len(judged_trigger),
        "Not Triggered Count": len(judged_not_trigger),
        "Expected Trigger Count (Bot Accuracy > Phase1 Accuracy + Margin)": len(expected_trigger),
        "Expected Not Trigger Count (Bot Accuracy <= Phase1 Accuracy + Margin)": len(expected_not_trigger),
        "False Positive Count (triggered but Bot Accuracy did not improve Phase1)": len(false_positive),
        "False Negative Count (not triggered but Bot Accuracy would improve Phase1)": len(false_negative),
        "Judgment Error Count": error_count,
        "Judgment Error Rate": error_count / total_scored if total_scored else 0.0,
        "Judgment Accuracy (1 - Judgment Error Rate)": (
            1.0 - (error_count / total_scored) if total_scored else 0.0
        ),
        "Average Phase1 Accuracy": (
            sum(j["phase1_accuracy"] for j in scored) / total_scored
            if total_scored else 0.0
        ),
        "Average Bot Accuracy": (
            sum(j["bot_accuracy"] for j in scored) / total_scored
            if total_scored else 0.0
        ),
        "Average Accuracy Gain (Bot - Phase1)": (
            sum(j["accuracy_gain"] for j in scored) / total_scored
            if total_scored else 0.0
        ),
    }


def fallback_miss_gain_summary(
    records: list[dict],
    provider_name: str | None = None,
    margin: float = 0.0,
    logger=None,
) -> dict:
    judgments = fallback_gain_judgments(
        records,
        provider_name=provider_name,
        margin=margin,
        logger=logger,
    )
    scored = [j for j in judgments if j["scored"]]
    not_triggered = [j for j in scored if not j["judged_trigger"]]
    misses = [
        j for j in not_triggered
        if j["expected_trigger"]
    ]
    total_records = len(records)
    return {
        "Total Records": total_records,
        "Total Scored Records (records with both Phase1 and Bot Accuracy)": len(scored),
        "Unscored Count (missing Phase1 or Bot Accuracy)": len(judgments) - len(scored),
        "Miss Count (not triggered and Bot Accuracy > Phase1 Accuracy + Margin)": len(misses),
        "Miss Rate Overall (Miss Count / Total Records)": (
            len(misses) / total_records if total_records else 0.0
        ),
        "Miss Rate Among Scored Records (Miss Count / Scored Records)": (
            len(misses) / len(scored) if scored else 0.0
        ),
        "Miss Rate Among Not Triggered (Miss Count / Not Triggered Count)": (
            len(misses) / len(not_triggered) if not_triggered else 0.0
        ),
    }


def recoverable_miss_gain_summary(
    records: list[dict],
    provider_name: str | None = None,
    margin: float = 0.0,
    logger=None,
) -> dict:
    judgments = fallback_gain_judgments(
        records,
        provider_name=provider_name,
        margin=margin,
        logger=logger,
    )
    scored = [j for j in judgments if j["scored"]]
    recoverable = [
        j for j in scored
        if (
            (not j["judged_trigger"])
            and j["accuracy_gain"] > 1.0
        )
    ]
    total_records = len(records)
    avg_gain = (
        sum(j["accuracy_gain"] for j in recoverable) / len(recoverable)
        if recoverable else 0.0
    )
    return {
        "Total Records": total_records,
        "Total Scored Records (records with both Phase1 and Bot Accuracy)": len(scored),
        "Unscored Count (missing Phase1 or Bot Accuracy)": len(judgments) - len(scored),
        "Recoverable Miss Count (not triggered, Bot Accuracy > Phase1 Accuracy + Margin)": len(recoverable),
        "Recoverable Miss Rate Overall (Recoverable Miss Count / Total Records)": (
            len(recoverable) / total_records if total_records else 0.0
        ),
        "Recoverable Miss Rate Among Scored Records (Recoverable Miss Count / Scored Records)": (
            len(recoverable) / len(scored) if scored else 0.0
        ),
        "Average Gain Among Recoverable Misses": avg_gain,
    }
