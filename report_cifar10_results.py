"""Build a CIFAR-10-only report from every log under outputs/.

The report combines:
  * CIFAR/CIFAR-10-named run directories under outputs/<date>/<run>/.
  * CIFAR-10 rows from outputs/sweep_summary_*.csv.
  * Known legacy CIFAR-10 FedAvg baseline directories whose names do not contain
    the dataset token.

Outputs:
  * cifar10_results_report.csv - detailed run-level table.
  * cifar10_results_report.md  - compact human-readable summary.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
REPORT_CSV = ROOT / "cifar10_results_report.csv"
REPORT_MD = ROOT / "cifar10_results_report.md"

CIFAR_RE = re.compile(r"(?:^|_)cifar(?:10)?(?:_|$)", re.IGNORECASE)
TIMESTAMP_PREFIX = re.compile(r"^\d{2}-\d{2}-\d{2}_")
TAU_RE = re.compile(r"(?:^|_)tau(?P<tau>\d+(?:\.\d+)?)(?:_|$)", re.IGNORECASE)
SUMMARY_TS_RE = re.compile(r"sweep_summary_(\d{4})(\d{2})(\d{2})_(\d{6})\.csv$")

ATTACK_LABELS = {
    "gaussian": "Gaussian",
    "lie": "LIE",
    "min_max": "Min-Max",
    "fang": "Fang",
    "no_attack": "No Attack",
    "none": "No Attack",
}

METHOD_ORDER = {
    "FedAvg": 0,
    "FLTrust": 1,
    "FLDetector": 2,
    "Multi-Krum": 3,
    "Our Method": 4,
}

# These old FedAvg baseline run directories are CIFAR-10 runs, but their run names
# predate the dataset token convention.
LEGACY_CIFAR_RUNS = {
    Path("outputs/2026-04-02/18-41-29_no_attack"): {
        "method": "FedAvg",
        "attack_mode": "no_attack",
    },
    Path("outputs/2026-04-02/18-53-21_gaussian"): {
        "method": "FedAvg",
        "attack_mode": "gaussian",
    },
    Path("outputs/2026-04-02/19-08-43_lie"): {
        "method": "FedAvg",
        "attack_mode": "lie",
    },
    Path("outputs/2026-04-02/19-35-26_min_max"): {
        "method": "FedAvg",
        "attack_mode": "min_max",
    },
}

FIELDNAMES = [
    "date",
    "run",
    "method",
    "attack_mode",
    "attack_label",
    "tau_s_quantile",
    "config_tau_s_quantile",
    "config_attack_mode",
    "config_rounds",
    "local_epochs",
    "actual_rounds",
    "final_round",
    "final_accuracy",
    "best_accuracy",
    "best_accuracy_round",
    "final_server_loss",
    "final_client_accuracy",
    "final_client_loss",
    "total_client_records",
    "malicious_records",
    "benign_records",
    "true_positive",
    "false_negative",
    "false_positive",
    "true_negative",
    "asr_all_clients",
    "fdr_all_clients",
    "detection_recall",
    "attack_miss_rate",
    "false_positive_rate",
    "total_convlstm_forwards",
    "mean_ratio_scored",
    "total_convlstm_time_ms",
    "status",
    "source",
    "sweep_summary_files",
    "error",
    "run_dir",
    "results_path",
]


def normalize_path(path_text: str | Path) -> Path:
    """Return a path relative to this repository when possible."""
    path = Path(path_text)
    if path.is_absolute():
        try:
            return path.relative_to(ROOT)
        except ValueError:
            return path
    return path


def rel_text(path: Path | str) -> str:
    return normalize_path(path).as_posix()


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def parse_int(value: Any) -> int | None:
    number = parse_float(value)
    if number is None:
        return None
    return int(number)


def fmt_float(value: Any, digits: int = 4) -> str:
    number = parse_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def fmt_int(value: Any) -> str:
    number = parse_int(value)
    if number is None:
        return ""
    return str(number)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_run_config(run_dir: Path) -> dict[str, Any]:
    path = ROOT / run_dir / "run_config.json"
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def round_sort_value(entry: dict[str, Any]) -> int | float:
    round_number = entry.get("round", 0)
    try:
        return int(round_number)
    except (TypeError, ValueError):
        try:
            return float(round_number)
        except (TypeError, ValueError):
            return 0


def normalise_result_entries(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        entries = [entry for entry in data if isinstance(entry, dict)]
        return sorted(entries, key=round_sort_value)

    if isinstance(data, dict):
        entries = []
        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            entry = dict(value)
            entry.setdefault("round", int(key) if str(key).isdigit() else key)
            entries.append(entry)
        return sorted(entries, key=round_sort_value)

    return []


def nested_metrics(entry: dict[str, Any], key: str) -> dict[str, Any]:
    value = entry.get(key)
    return value if isinstance(value, dict) else {}


def extract_accuracy(entry: dict[str, Any]) -> float | None:
    server_metrics = nested_metrics(entry, "evaluate_metrics_serverapp")
    if "accuracy" in server_metrics:
        return parse_float(server_metrics["accuracy"])

    client_metrics = nested_metrics(entry, "evaluate_metrics_clientapp")
    if "eval_acc" in client_metrics:
        return parse_float(client_metrics["eval_acc"])

    for key in ("accuracy", "eval_accuracy"):
        if key in entry:
            return parse_float(entry[key])

    return None


def extract_results_summary(results_path: Path) -> tuple[dict[str, str], str]:
    """Return result metrics and status for a results.json file."""
    try:
        data = read_json(results_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"invalid_results: {exc}"

    entries = normalise_result_entries(data)
    if not entries:
        return {"actual_rounds": "0"}, "no_rounds"

    final_entry = entries[-1]
    final_accuracy = extract_accuracy(final_entry)
    best_entry = max(
        (entry for entry in entries if extract_accuracy(entry) is not None),
        key=lambda entry: extract_accuracy(entry) or -math.inf,
        default=None,
    )

    server_metrics = nested_metrics(final_entry, "evaluate_metrics_serverapp")
    client_metrics = nested_metrics(final_entry, "evaluate_metrics_clientapp")

    result = {
        "actual_rounds": str(len(entries)),
        "final_round": str(final_entry.get("round", "")),
        "final_accuracy": fmt_float(final_accuracy),
        "final_server_loss": fmt_float(server_metrics.get("loss"), digits=6),
        "final_client_accuracy": fmt_float(client_metrics.get("eval_acc")),
        "final_client_loss": fmt_float(client_metrics.get("eval_loss"), digits=6),
    }

    if best_entry is not None:
        result["best_accuracy"] = fmt_float(extract_accuracy(best_entry))
        result["best_accuracy_round"] = str(best_entry.get("round", ""))

    status = "ok" if final_accuracy is not None else "no_accuracy"
    return result, status


def detect_attack(run_name: str) -> str:
    text = f"_{TIMESTAMP_PREFIX.sub('', run_name).lower()}_"
    if "_no_attack_" in text:
        return "no_attack"
    if "_none_" in text:
        return "no_attack"
    if "_min_max_" in text or "_minmax_" in text:
        return "min_max"
    for attack in ("gaussian", "lie", "fang"):
        if f"_{attack}_" in text:
            return attack
    return ""


def detect_method(run_name: str) -> str:
    text = f"_{TIMESTAMP_PREFIX.sub('', run_name).lower()}_"
    if "_fltrust_" in text:
        return "FLTrust"
    if "_fldetector_" in text:
        return "FLDetector"
    if "_multikrum_" in text:
        return "Multi-Krum"
    if "_fedavg_" in text:
        return "FedAvg"
    return "Our Method"


def detect_tau(run_name: str) -> str:
    match = TAU_RE.search(TIMESTAMP_PREFIX.sub("", run_name))
    if match is None:
        return ""
    return fmt_float(match.group("tau"))


def parse_run_identity(
    run_dir: Path,
    config: dict[str, Any],
    summary: dict[str, Any] | None = None,
) -> dict[str, str]:
    legacy = LEGACY_CIFAR_RUNS.get(run_dir, {})
    run_name = run_dir.name
    method = legacy.get("method") or detect_method(run_name)
    attack = (
        legacy.get("attack_mode")
        or detect_attack(run_name)
        or str((summary or {}).get("attack_mode", "")).strip()
        or str(config.get("attack-mode", "")).strip()
    )
    tau = (
        legacy.get("tau_s_quantile", "")
        or detect_tau(run_name)
        or fmt_float((summary or {}).get("tau_s_quantile"))
        or fmt_float(config.get("tau-s-quantile"))
    )

    return {
        "method": method,
        "attack_mode": attack,
        "attack_label": ATTACK_LABELS.get(attack, attack.replace("_", " ").title()),
        "tau_s_quantile": tau,
    }


def path_date(run_dir: Path) -> str:
    parts = run_dir.parts
    if len(parts) >= 3 and parts[0] == "outputs":
        return parts[1]
    return ""


def read_detection_summary(run_dir: Path) -> dict[str, str]:
    path = ROOT / run_dir / "client_states_rounds.csv"
    if not path.exists():
        return {}

    tp = fn = fp = tn = total = malicious = benign = 0
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            is_malicious = parse_int(row.get("is_malicious"))
            prediction = parse_int(row.get("predict_state"))
            if is_malicious is None or prediction is None:
                continue

            total += 1
            if is_malicious == 1:
                malicious += 1
                if prediction == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                benign += 1
                if prediction == 1:
                    fp += 1
                else:
                    tn += 1

    if total == 0:
        return {}

    return {
        "total_client_records": str(total),
        "malicious_records": str(malicious),
        "benign_records": str(benign),
        "true_positive": str(tp),
        "false_negative": str(fn),
        "false_positive": str(fp),
        "true_negative": str(tn),
        "asr_all_clients": fmt_float(fn / total),
        "fdr_all_clients": fmt_float(fp / total),
        "detection_recall": fmt_float(tp / malicious if malicious else None),
        "attack_miss_rate": fmt_float(fn / malicious if malicious else None),
        "false_positive_rate": fmt_float(fp / benign if benign else None),
    }


def read_cost_summary(run_dir: Path) -> dict[str, str]:
    path = ROOT / run_dir / "convlstm_cost.csv"
    if not path.exists():
        return {}

    forwards = 0
    ratio_sum = 0.0
    ratio_count = 0
    total_time_ms = 0.0

    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scored = parse_int(row.get("n_scored"))
            if scored is not None:
                forwards += scored
            ratio = parse_float(row.get("ratio_scored"))
            if ratio is not None:
                ratio_sum += ratio
                ratio_count += 1
            elapsed = parse_float(row.get("t_convlstm"))
            if elapsed is not None:
                total_time_ms += elapsed * 1000.0

    result = {"total_convlstm_forwards": str(forwards)}
    if ratio_count:
        result["mean_ratio_scored"] = fmt_float(ratio_sum / ratio_count)
    result["total_convlstm_time_ms"] = fmt_float(total_time_ms, digits=2)
    return result


def read_sweep_summaries() -> tuple[dict[Path, list[dict[str, str]]], list[dict[str, str]]]:
    by_path: dict[Path, list[dict[str, str]]] = defaultdict(list)
    no_path_rows: list[dict[str, str]] = []

    for summary_path in sorted(OUTPUTS_DIR.glob("sweep_summary_*.csv")):
        with open(summary_path, "r", newline="", encoding="utf-8") as f:
            for row_index, row in enumerate(csv.DictReader(f), start=2):
                if row.get("dataset") != "cifar10":
                    continue

                row = dict(row)
                row["_summary_file"] = rel_text(summary_path)
                row["_summary_row"] = str(row_index)
                save_path = row.get("save_path", "").strip()
                if save_path:
                    by_path[normalize_path(save_path)].append(row)
                else:
                    no_path_rows.append(row)

    return by_path, no_path_rows


def discover_run_dirs(summary_by_path: dict[Path, list[dict[str, str]]]) -> list[Path]:
    run_dirs = {
        normalize_path(path)
        for path in OUTPUTS_DIR.glob("*/*")
        if path.is_dir() and CIFAR_RE.search(path.name)
    }
    run_dirs.update(path for path in LEGACY_CIFAR_RUNS if (ROOT / path).exists())
    run_dirs.update(summary_by_path)
    return sorted(run_dirs, key=lambda p: p.as_posix())


def latest_summary(summaries: list[dict[str, str]]) -> dict[str, str] | None:
    if not summaries:
        return None
    return summaries[-1]


def apply_summary_fallbacks(row: dict[str, str], summary: dict[str, str] | None) -> None:
    if summary is None:
        return

    for src, dst, digits in (
        ("final_accuracy", "final_accuracy", 4),
        ("total_convlstm_forwards", "total_convlstm_forwards", 0),
        ("mean_ratio_scored", "mean_ratio_scored", 4),
        ("total_convlstm_time_ms", "total_convlstm_time_ms", 2),
    ):
        if row.get(dst):
            continue
        value = parse_float(summary.get(src))
        if value is None:
            continue
        row[dst] = str(int(value)) if digits == 0 else fmt_float(value, digits=digits)


def build_run_row(
    run_dir: Path,
    summaries: list[dict[str, str]],
) -> dict[str, str]:
    summary = latest_summary(summaries)
    config = read_run_config(run_dir)
    identity = parse_run_identity(run_dir, config, summary)
    results_path = run_dir / "results.json"
    absolute_results_path = ROOT / results_path

    source_parts = []
    if run_dir in LEGACY_CIFAR_RUNS:
        source_parts.append("legacy_curation")
    elif (ROOT / run_dir).exists():
        source_parts.append("run_dir_name")
    else:
        source_parts.append("sweep_summary")
    if summaries and "sweep_summary" not in source_parts:
        source_parts.append("sweep_summary")

    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "date": path_date(run_dir),
            "run": run_dir.name,
            "run_dir": rel_text(run_dir),
            "results_path": rel_text(results_path),
            "config_tau_s_quantile": fmt_float(config.get("tau-s-quantile")),
            "config_attack_mode": str(config.get("attack-mode", "")),
            "config_rounds": fmt_int(config.get("num-server-rounds")),
            "local_epochs": fmt_int(config.get("local-epochs")),
            "source": ";".join(source_parts),
            "sweep_summary_files": ";".join(
                row["_summary_file"] for row in summaries if row.get("_summary_file")
            ),
            "error": "; ".join(
                row["error"] for row in summaries if row.get("error")
            ),
            **identity,
        }
    )

    if absolute_results_path.exists():
        result_values, status = extract_results_summary(absolute_results_path)
        row.update(result_values)
        row["status"] = status
    elif (ROOT / run_dir).exists():
        row["status"] = "missing_results_summary" if summary else "missing_results"
    else:
        row["status"] = "summary_missing_dir"

    row.update(read_detection_summary(run_dir))
    row.update(read_cost_summary(run_dir))
    apply_summary_fallbacks(row, summary)

    if row["status"] in {"missing_results_summary", "summary_missing_dir"}:
        if not row.get("final_accuracy"):
            row["status"] = f"{row['status']}_no_accuracy"

    return row


def summary_date(summary_file: str) -> str:
    match = SUMMARY_TS_RE.search(Path(summary_file).name)
    if match is None:
        return ""
    year, month, day, _time = match.groups()
    return f"{year}-{month}-{day}"


def build_summary_only_row(summary: dict[str, str]) -> dict[str, str]:
    config: dict[str, Any] = {}
    identity = parse_run_identity(Path(""), config, summary)

    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "date": summary_date(summary.get("_summary_file", "")),
            "method": "Our Method",
            "attack_mode": summary.get("attack_mode", ""),
            "attack_label": ATTACK_LABELS.get(
                summary.get("attack_mode", ""),
                summary.get("attack_mode", "").replace("_", " ").title(),
            ),
            "tau_s_quantile": fmt_float(summary.get("tau_s_quantile")),
            "final_accuracy": fmt_float(summary.get("final_accuracy")),
            "total_convlstm_forwards": fmt_int(summary.get("total_convlstm_forwards")),
            "mean_ratio_scored": fmt_float(summary.get("mean_ratio_scored")),
            "total_convlstm_time_ms": fmt_float(
                summary.get("total_convlstm_time_ms"),
                digits=2,
            ),
            "status": "summary_error" if summary.get("error") else "summary_only",
            "source": "sweep_summary",
            "sweep_summary_files": summary.get("_summary_file", ""),
            "error": summary.get("error", ""),
        }
    )
    row.update({key: value for key, value in identity.items() if value})
    return row


def numeric_accuracy(row: dict[str, str]) -> float | None:
    return parse_float(row.get("final_accuracy"))


def tau_sort_value(row: dict[str, str]) -> float:
    tau = parse_float(row.get("tau_s_quantile"))
    return tau if tau is not None else math.inf


def sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get("date", ""),
            row.get("run", ""),
            row.get("attack_mode", ""),
            tau_sort_value(row),
            row.get("sweep_summary_files", ""),
        ),
    )


def write_csv(rows: list[dict[str, str]]) -> None:
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def tau_values_text(rows: list[dict[str, str]]) -> str:
    values = sorted({parse_float(row.get("tau_s_quantile")) for row in rows})
    values = [value for value in values if value is not None]
    if not values:
        return ""
    if len(values) > 6:
        return f"{values[0]:.2f}-{values[-1]:.2f} ({len(values)} values)"
    return ", ".join(f"{value:.2f}" for value in values)


def best_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    numeric_rows = [row for row in rows if numeric_accuracy(row) is not None]
    if not numeric_rows:
        return None
    return max(numeric_rows, key=lambda row: numeric_accuracy(row) or -math.inf)


def write_markdown(rows: list[dict[str, str]]) -> None:
    status_counts = Counter(row["status"] for row in rows)
    source_counts = Counter(row["source"] for row in rows)
    numeric_rows = [row for row in rows if numeric_accuracy(row) is not None]

    lines = [
        "# CIFAR-10 Results Report",
        "",
        f"Detailed CSV: `{REPORT_CSV.name}`",
        "",
        "Scope: all CIFAR/CIFAR-10-named run directories under `outputs/`, "
        "CIFAR-10 rows from `outputs/sweep_summary_*.csv`, and the known "
        "legacy unlabeled CIFAR-10 FedAvg baseline directories from 2026-04-02.",
        "",
        "## Counts",
        "",
    ]
    lines.extend(
        markdown_table(
            ["Metric", "Value"],
            [
                ["Report rows", str(len(rows))],
                ["Rows with numeric final accuracy", str(len(numeric_rows))],
                ["Rows with detection logs", str(sum(1 for row in rows if row["total_client_records"]))],
                ["Rows with ConvLSTM forwards", str(sum(1 for row in rows if row["total_convlstm_forwards"]))],
            ],
        )
    )

    lines.extend(["", "## Status Counts", ""])
    lines.extend(
        markdown_table(
            ["Status", "Rows"],
            [[status, str(count)] for status, count in sorted(status_counts.items())],
        )
    )

    lines.extend(["", "## Source Counts", ""])
    lines.extend(
        markdown_table(
            ["Source", "Rows"],
            [[source, str(count)] for source, count in sorted(source_counts.items())],
        )
    )

    by_attack: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_attack[row["attack_mode"]].append(row)

    coverage_rows = []
    for attack, attack_rows in sorted(by_attack.items()):
        best = best_row(attack_rows)
        coverage_rows.append(
            [
                ATTACK_LABELS.get(attack, attack),
                str(len(attack_rows)),
                str(sum(1 for row in attack_rows if numeric_accuracy(row) is not None)),
                tau_values_text(attack_rows),
                best["final_accuracy"] if best else "",
                best["run"] if best else "",
            ]
        )

    lines.extend(["", "## Coverage By Attack", ""])
    lines.extend(
        markdown_table(
            ["Attack", "Rows", "Rows With Accuracy", "Tau Values", "Best Accuracy", "Best Run"],
            coverage_rows,
        )
    )

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in numeric_rows:
        grouped[(row["attack_mode"], row["method"])].append(row)

    best_rows = []
    for (attack, method), group_rows in sorted(
        grouped.items(),
        key=lambda item: (
            ATTACK_LABELS.get(item[0][0], item[0][0]),
            METHOD_ORDER.get(item[0][1], 99),
        ),
    ):
        best = best_row(group_rows)
        if best is None:
            continue
        best_rows.append(
            [
                ATTACK_LABELS.get(attack, attack),
                method,
                best["final_accuracy"],
                best["tau_s_quantile"],
                best["final_round"],
                best["status"],
                best["run"],
            ]
        )

    lines.extend(["", "## Best Final Accuracy By Attack And Method", ""])
    lines.extend(
        markdown_table(
            ["Attack", "Method", "Final Accuracy", "Tau", "Final Round", "Status", "Run"],
            best_rows,
        )
    )

    incomplete = [row for row in rows if row["status"] != "ok"]
    lines.extend(["", "## Incomplete Or Summary-Only Logs", ""])
    if incomplete:
        lines.extend(
            markdown_table(
                ["Status", "Attack", "Tau", "Final Accuracy", "Run", "Error"],
                [
                    [
                        row["status"],
                        ATTACK_LABELS.get(row["attack_mode"], row["attack_mode"]),
                        row["tau_s_quantile"],
                        row["final_accuracy"],
                        row["run"],
                        row["error"].replace("|", "/"),
                    ]
                    for row in incomplete
                ],
            )
        )
    else:
        lines.append("No incomplete or summary-only CIFAR-10 logs found.")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary_by_path, summary_only = read_sweep_summaries()
    run_dirs = discover_run_dirs(summary_by_path)

    rows = [
        build_run_row(run_dir, summary_by_path.get(run_dir, []))
        for run_dir in run_dirs
    ]
    rows.extend(build_summary_only_row(row) for row in summary_only)
    rows = sort_rows(rows)

    write_csv(rows)
    write_markdown(rows)
    print(f"Saved -> {rel_text(REPORT_CSV)}")
    print(f"Saved -> {rel_text(REPORT_MD)}")
    print(
        f"CIFAR-10 report rows: {len(rows)} "
        f"({sum(1 for row in rows if numeric_accuracy(row) is not None)} with accuracy)"
    )


if __name__ == "__main__":
    main()
