"""Plot round-vs-accuracy curves and final-accuracy tables.

The script reads every results.json listed in ``JSON_FILES``.  Each top-level key is
a dataset, and each nested key is the curve/table label for that dataset.
"""

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# Configure: list every curated results.json you want to compare, grouped by dataset.
BASE_JSON_FILES = {
    "MNIST": {
        "FedAvg + Min-Max": "outputs/2026-04-01/fedavg_min_max/results.json",
        "FedAvg + LIE": "outputs/2026-04-01/fedavg_lie/results.json",
        "FedAvg + Gaussian": "outputs/2026-04-01/gaussian/results.json",
        "FedAvg + Fang": "outputs/2026-04-10/15-45-53_fang_FedAvg_mnist/results.json",
        "Our Method (Gaussian)": "outputs/2026-04-08/21-11-36_gaussian_mnist/results.json",
        "FLTrust (Gaussian)": "outputs/2026-04-08/15-15-33_gaussian_FLtrust_mnist/results.json",
        "FLDetector (Gaussian)": "outputs/2026-04-09/22-33-56_gaussian_FLdetector_mnist/results.json",
        "Multi-Krum (Gaussian)": "outputs/2026-04-08/18-25-18_gaussian_multikrum_mnist/results.json",
    },
    "Fashion-MNIST": {
        "FedAvg + Min-Max": "outputs/2026-04-02/18-02-27_min_max/results.json",
        "FedAvg + LIE": "outputs/2026-04-02/17-49-32_lie/results.json",
        "FedAvg + Gaussian": "outputs/2026-04-02/17-35-06_gaussian/results.json",
        "FedAvg + Fang": "outputs/2026-04-10/16-00-36_fang_FedAvg_famnist/results.json",
        "Our Method (Gaussian)": "outputs/2026-04-09/16-16-15_gaussian_famnist/results.json",
        "FLTrust (Gaussian)": "outputs/2026-04-09/19-42-14_gaussian_FLtrust_famnist/results.json",
        "FLDetector (Gaussian)": "outputs/2026-04-10/11-46-15_gaussian_FLdetector_famnist/results.json",
        "Multi-Krum (Gaussian)": "outputs/2026-04-09/16-29-46_gaussian_multikrum_famnist/results.json",
    },
}

# Also include every run from these output dates.
RESULT_DATES = (
    "2026-04-06",
    "2026-04-07",
    "2026-04-08",
    "2026-04-09",
    "2026-04-10",
    "2026-07-06",
    "2026-07-07",
    "2026-07-08",
    "2026-07-09",
    "2026-07-10",
    "2026-07-11",
    "2026-07-24",
    "2026-07-25",
)
INCLUDE_NO_ATTACK_DATES = {
    "2026-07-06",
    "2026-07-07",
    "2026-07-08",
    "2026-07-09",
    "2026-07-10",
    "2026-07-11",
}

DATASET_LABELS = {
    "mnist": "MNIST",
    "famnist": "Fashion-MNIST",
    "fashion_mnist": "Fashion-MNIST",
}

ATTACK_LABELS = {
    "gaussian": "Gaussian",
    "lie": "LIE",
    "minmax": "Min-Max",
    "min_max": "Min-Max",
    "fang": "Fang",
    "none": "No Attack",
    "no_attack": "No Attack",
}

METHOD_LABELS = {
    "FedAvg": "FedAvg",
    "FLtrust": "FLTrust",
    "FLdetector": "FLDetector",
    "multikrum": "Multi-Krum",
}

TIMESTAMP_PREFIX = re.compile(r"^\d{2}-\d{2}-\d{2}_")
TAU_TOKEN = re.compile(r"^tau(?P<value>\d+(?:\.\d+)?)$")
EXCLUDED_ATTACK_KEYS = {"none", "no_attack"}

OUT_DIR = Path(".")
TABLE_CSV = OUT_DIR / "final_accuracy_by_dataset.csv"
TABLE_MD = OUT_DIR / "final_accuracy_by_dataset.md"
ATTACK_DEFENSE_CSV = OUT_DIR / "final_accuracy_by_attack_defense_dataset.csv"
ATTACK_DEFENSE_MD = OUT_DIR / "final_accuracy_by_attack_defense_dataset.md"
SHOW_PLOTS = False

REQUESTED_ATTACKS = ("Min-Max", "Fang", "Gaussian", "LIE")
REQUESTED_DEFENSES = ("Multi-Krum", "FLTrust", "FLDefender", "Our Method")
REQUESTED_DATASETS = ("MNIST", "Fashion-MNIST")
LABEL_DATE_SUFFIX = re.compile(r" \(\d{4}-\d{2}-\d{2}(?: \d{2}-\d{2}-\d{2})?\)$")


def split_dataset(run_name: str) -> tuple[list[str], str] | None:
    """Split a run name into non-dataset tokens and a dataset label."""
    parts = run_name.split("_")
    if len(parts) >= 2 and "_".join(parts[-2:]) in DATASET_LABELS:
        return parts[:-2], DATASET_LABELS["_".join(parts[-2:])]

    if parts and parts[-1] in DATASET_LABELS:
        return parts[:-1], DATASET_LABELS[parts[-1]]

    if len(parts) >= 2 and "_".join(parts[:2]) in DATASET_LABELS:
        return parts[2:], DATASET_LABELS["_".join(parts[:2])]

    if parts and parts[0] in DATASET_LABELS:
        return parts[1:], DATASET_LABELS[parts[0]]

    if parts and parts[-1] in METHOD_LABELS:
        method_token = parts[-1]
        prefix_parts = parts[:-1]
        if len(prefix_parts) >= 2 and "_".join(prefix_parts[-2:]) in DATASET_LABELS:
            return (
                prefix_parts[:-2] + [method_token],
                DATASET_LABELS["_".join(prefix_parts[-2:])],
            )

        if prefix_parts and prefix_parts[-1] in DATASET_LABELS:
            return (
                prefix_parts[:-1] + [method_token],
                DATASET_LABELS[prefix_parts[-1]],
            )

    return None


def split_tau(tokens: list[str]) -> tuple[list[str], str | None]:
    """Remove tau tokens from run-name tokens and return the tau value."""
    attack_tokens = []
    tau = None

    for token in tokens:
        match = TAU_TOKEN.fullmatch(token.lower())
        if match:
            tau = match.group("value")
        else:
            attack_tokens.append(token)

    return attack_tokens, tau


def build_label(method: str, attack: str, tau: str | None = None) -> str:
    """Return a plot/table label for one method and attack pair."""
    if tau is not None:
        attack = f"{attack}, tau={tau}"

    if method == "FedAvg":
        if tau is None:
            return f"FedAvg + {attack}"
        return f"FedAvg + {attack.split(', ', maxsplit=1)[0]} (tau={tau})"

    return f"{method} ({attack})"


def label_from_result_path(path: Path) -> tuple[str, str] | None:
    """Infer (dataset, label) from an outputs/<date>/<run>/results.json path."""
    run_name = TIMESTAMP_PREFIX.sub("", path.parent.name)
    split = split_dataset(run_name)
    if split is None:
        return None

    tokens, dataset = split
    method = "Our Method"
    for method_token, method_label in METHOD_LABELS.items():
        if tokens and tokens[-1] == method_token:
            method = method_label
            tokens = tokens[:-1]
            break

    tokens, tau = split_tau(tokens)
    attack_tokens = [
        token
        for token in tokens
        if not token.lower().startswith(("lamda", "lambda"))
    ]
    attack_key = "_".join(attack_tokens)
    if attack_key in EXCLUDED_ATTACK_KEYS:
        date = path.parts[1] if len(path.parts) > 1 else ""
        if date not in INCLUDE_NO_ATTACK_DATES:
            return None

    attack = ATTACK_LABELS.get(attack_key)
    if attack is None:
        attack = attack_key.replace("_", " ").title()

    return dataset, build_label(method, attack, tau)


def unique_label(entries: dict[str, str], label: str, path: Path) -> str:
    """Return a stable label that does not overwrite a different configured path."""
    if label not in entries:
        return label

    date = path.parts[1] if len(path.parts) > 1 else path.parent.name
    candidate = f"{label} ({date})"
    if candidate not in entries:
        return candidate

    run_time = path.parent.name[:8]
    suffix = f"{date} {run_time}"
    return f"{label} ({suffix})"


def add_date_results(
    base_json_files: dict[str, dict[str, str]],
    dates: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    """Append every results.json from selected output dates to configured results."""
    json_files = {
        dataset: dict(entries)
        for dataset, entries in base_json_files.items()
    }

    for date in dates:
        date_dir = Path("outputs") / date
        if not date_dir.exists():
            continue

        for path in sorted(date_dir.glob("*/results.json")):
            label_info = label_from_result_path(path)
            if label_info is None:
                continue

            dataset, label = label_info
            entries = json_files.setdefault(dataset, {})
            path_str = str(path)
            if path_str in entries.values():
                continue

            entries[unique_label(entries, label, path)] = path_str

    return json_files


JSON_FILES = add_date_results(BASE_JSON_FILES, RESULT_DATES)


def dataset_slug(dataset: str) -> str:
    """Return a filesystem-friendly dataset name."""
    return (
        dataset.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def normalise_result_entries(data: Any) -> list[dict[str, Any]]:
    """Return result entries from current list and legacy dict JSON shapes."""
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]

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


def round_sort_value(entry: dict[str, Any]) -> int | float:
    """Return a stable numeric sort key for round-like values."""
    round_number = entry.get("round", 0)
    try:
        return int(round_number)
    except (TypeError, ValueError):
        return 0


def extract_accuracy(entry: dict[str, Any]) -> float | None:
    """Extract accuracy from current and legacy results.json metric names."""
    server_metrics = entry.get("evaluate_metrics_serverapp") or {}
    if "accuracy" in server_metrics:
        return float(server_metrics["accuracy"])

    client_metrics = entry.get("evaluate_metrics_clientapp") or {}
    if "eval_acc" in client_metrics:
        return float(client_metrics["eval_acc"])

    for key in ("accuracy", "eval_accuracy"):
        if key in entry:
            return float(entry[key])

    return None


def load_rounds_and_accuracy(path: str) -> tuple[list[int | float], list[float]]:
    """Return (rounds, accuracies) lists from a results.json file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rounds, accs = [], []
    for index, entry in enumerate(normalise_result_entries(data), start=1):
        round_number = entry.get("round", index)
        accuracy = extract_accuracy(entry)
        if round_number is not None and accuracy is not None:
            rounds.append(round_number)
            accs.append(accuracy)

    return rounds, accs


def build_row(dataset: str, label: str, path: str) -> dict[str, Any]:
    """Build one final-accuracy summary row for a configured result file."""
    result_path = Path(path)
    row = {
        "dataset": dataset,
        "label": label,
        "final_round": "",
        "final_accuracy": "",
        "path": path,
        "status": "ok",
    }

    if not result_path.exists():
        row["status"] = "missing"
        return row

    rounds, accs = load_rounds_and_accuracy(path)
    if not accs:
        row["status"] = "no_accuracy"
        return row

    row["final_round"] = rounds[-1]
    row["final_accuracy"] = accs[-1]
    return row


def plot_dataset(dataset: str, json_files: dict[str, str]) -> list[dict[str, Any]]:
    """Plot all configured curves for one dataset and return summary rows."""
    fig_height = max(6, min(18, 0.24 * max(len(json_files), 1)))
    legend_font = 8 if len(json_files) > 35 else 10
    fig, ax = plt.subplots(figsize=(10, fig_height))
    rows = []
    plotted_any = False

    for label, path in json_files.items():
        row = build_row(dataset, label, path)
        rows.append(row)

        if row["status"] != "ok":
            print(f"[SKIP] {dataset} / {label}: {row['status']} ({path})")
            continue

        rounds, accs = load_rounds_and_accuracy(path)
        ax.plot(rounds, accs, linewidth=2.5, label=label)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print(f"[SKIP] No plottable results for {dataset}")
        return rows

    ax.set_title(f"{dataset}: Accuracy vs Round", fontsize=18)
    ax.set_xlabel("Round", fontsize=18)
    ax.set_ylabel("Accuracy", fontsize=18)
    ax.legend(fontsize=legend_font, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / f"accuracy_plot_{dataset_slug(dataset)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return rows


def ok_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows with a valid final accuracy, sorted best-to-worst per dataset."""
    return sorted(
        [row for row in rows if row["status"] == "ok"],
        key=lambda row: float(row["final_accuracy"]),
        reverse=True,
    )


def print_table(dataset: str, rows: list[dict[str, Any]]) -> None:
    """Print one final-accuracy comparison table."""
    valid_rows = ok_rows(rows)
    if not valid_rows:
        print(f"\n{dataset}: no final accuracy values found")
        return

    print(f"\n{dataset} final accuracy")
    print("-" * 72)
    print(f"{'Rank':>4}  {'Label':<32}  {'Round':>5}  {'Final Acc.':>10}")
    print("-" * 72)
    for rank, row in enumerate(valid_rows, start=1):
        print(
            f"{rank:>4}  "
            f"{row['label']:<32}  "
            f"{row['final_round']:>5}  "
            f"{float(row['final_accuracy']):>10.4f}"
        )

    skipped = [row for row in rows if row["status"] != "ok"]
    for row in skipped:
        print(f"[SKIP] {row['label']}: {row['status']} ({row['path']})")


def parse_attack_defense(label: str) -> tuple[str, str] | None:
    """Return (attack type, defense) from a plot/table label."""
    label = LABEL_DATE_SUFFIX.sub("", label)

    if label.startswith("FedAvg + "):
        return None

    for prefix, defense in (
        ("Multi-Krum", "Multi-Krum"),
        ("FLTrust", "FLTrust"),
        ("FLDetector", "FLDefender"),
        ("FLDefender", "FLDefender"),
        ("Our Method", "Our Method"),
    ):
        match = re.fullmatch(rf"{re.escape(prefix)} \((.+)\)", label)
        if match:
            attack = match.group(1).split(",", maxsplit=1)[0].strip()
            return attack, defense

    return None


def attack_defense_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build requested final-accuracy rows, keeping the best duplicate result."""
    best_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in rows:
        if row["status"] != "ok":
            continue

        parsed = parse_attack_defense(row["label"])
        if parsed is None:
            continue

        attack, defense = parsed
        if (
            attack not in REQUESTED_ATTACKS
            or defense not in REQUESTED_DEFENSES
            or row["dataset"] not in REQUESTED_DATASETS
        ):
            continue

        key = (attack, defense, row["dataset"])
        table_row = {
            "Attack type": attack,
            "Defensive": defense,
            "Dataset": row["dataset"],
            "Final accuracy": float(row["final_accuracy"]),
        }

        current = best_by_key.get(key)
        if current is None or table_row["Final accuracy"] > current["Final accuracy"]:
            best_by_key[key] = table_row

    return [
        best_by_key[(attack, defense, dataset)]
        for attack in REQUESTED_ATTACKS
        for defense in REQUESTED_DEFENSES
        for dataset in REQUESTED_DATASETS
        if (attack, defense, dataset) in best_by_key
    ]


def save_csv(rows: list[dict[str, Any]]) -> None:
    """Save all final-accuracy rows to CSV."""
    with open(TABLE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "label",
                "final_round",
                "final_accuracy",
                "status",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved -> {TABLE_CSV}")


def save_attack_defense_csv(rows: list[dict[str, Any]]) -> None:
    """Save requested attack/defense/dataset final-accuracy rows to CSV."""
    with open(ATTACK_DEFENSE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Attack type", "Defensive", "Dataset", "Final accuracy"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "Final accuracy": f"{row['Final accuracy']:.4f}",
                }
            )
    print(f"Saved -> {ATTACK_DEFENSE_CSV}")


def save_markdown(rows_by_dataset: dict[str, list[dict[str, Any]]]) -> None:
    """Save one Markdown section per dataset, ranked by final accuracy."""
    lines = ["# Final Accuracy by Dataset", ""]
    for dataset, rows in rows_by_dataset.items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Rank | Label | Final Round | Final Accuracy |",
                "|---:|---|---:|---:|",
            ]
        )

        valid_rows = ok_rows(rows)
        if valid_rows:
            for rank, row in enumerate(valid_rows, start=1):
                lines.append(
                    f"| {rank} | {row['label']} | {row['final_round']} | "
                    f"{float(row['final_accuracy']):.4f} |"
                )
        else:
            lines.append("|  | No valid results |  |  |")

        skipped = [row for row in rows if row["status"] != "ok"]
        if skipped:
            lines.extend(["", "Skipped:", ""])
            for row in skipped:
                lines.append(f"- {row['label']}: {row['status']} (`{row['path']}`)")

        lines.append("")

    TABLE_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved -> {TABLE_MD}")


def save_attack_defense_markdown(rows: list[dict[str, Any]]) -> None:
    """Save requested attack/defense/dataset final-accuracy rows to Markdown."""
    lines = [
        "# Final Accuracy by Attack, Defense, and Dataset",
        "",
        "| Attack type | Defensive | Dataset | Final accuracy |",
        "|---|---|---|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['Attack type']} | {row['Defensive']} | {row['Dataset']} | "
            f"{row['Final accuracy']:.4f} |"
        )

    ATTACK_DEFENSE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved -> {ATTACK_DEFENSE_MD}")


def main() -> None:
    rows_by_dataset = {}
    all_rows = []

    for dataset, json_files in JSON_FILES.items():
        rows = plot_dataset(dataset, json_files)
        rows_by_dataset[dataset] = rows
        all_rows.extend(rows)
        print_table(dataset, rows)

    save_csv(all_rows)
    save_markdown(rows_by_dataset)
    requested_rows = attack_defense_rows(all_rows)
    save_attack_defense_csv(requested_rows)
    save_attack_defense_markdown(requested_rows)


if __name__ == "__main__":
    main()
