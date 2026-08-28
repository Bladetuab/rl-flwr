# CIFAR-10 Results Report

Detailed CSV: `cifar10_results_report.csv`

Scope: all CIFAR/CIFAR-10-named run directories under `outputs/`, CIFAR-10 rows from `outputs/sweep_summary_*.csv`, and the known legacy unlabeled CIFAR-10 FedAvg baseline directories from 2026-04-02.

## Counts

| Metric | Value |
| --- | --- |
| Report rows | 156 |
| Rows with numeric final accuracy | 134 |
| Rows with detection logs | 90 |
| Rows with ConvLSTM forwards | 114 |

## Status Counts

| Status | Rows |
| --- | --- |
| missing_results | 2 |
| missing_results_summary | 4 |
| ok | 110 |
| summary_error | 20 |
| summary_missing_dir | 20 |

## Source Counts

| Source | Rows |
| --- | --- |
| legacy_curation | 4 |
| run_dir_name | 19 |
| run_dir_name;sweep_summary | 93 |
| sweep_summary | 40 |

## Coverage By Attack

| Attack | Rows | Rows With Accuracy | Tau Values | Best Accuracy | Best Run |
| --- | --- | --- | --- | --- | --- |
| Fang | 16 | 16 | 0.00-1.00 (11 values) | 0.5105 | 12-52-17_fang_FLdetector_cifar |
| Gaussian | 41 | 40 | 0.00-1.00 (11 values) | 0.6203 | 21-19-44_cifar10_gaussian_tau0.70 |
| LIE | 42 | 41 | 0.00-1.00 (11 values) | 0.5691 | 00-59-56_cifar10_lie_tau0.10 |
| Min-Max | 42 | 34 | 0.00-1.00 (11 values) | 0.5911 | 12-29-48_cifar10_min_max_tau0.50 |
| No Attack | 15 | 3 | 0.00-1.00 (11 values) | 0.6170 | 18-41-29_no_attack |

## Best Final Accuracy By Attack And Method

| Attack | Method | Final Accuracy | Tau | Final Round | Status | Run |
| --- | --- | --- | --- | --- | --- | --- |
| Fang | FedAvg | 0.4473 | 0.8000 | 30 | ok | 15-23-19_fang_FedAvg_cifar |
| Fang | FLTrust | 0.4914 | 0.8000 | 30 | ok | 18-45-02_fang_FLtrust_cifar |
| Fang | FLDetector | 0.5105 | 0.8000 | 30 | ok | 12-52-17_fang_FLdetector_cifar |
| Fang | Multi-Krum | 0.1000 | 0.8000 | 30 | ok | 19-23-35_fang_multikrum_cifar |
| Fang | Our Method | 0.3881 | 0.6000 | 50 | ok | 06-25-17_cifar10_fang_tau0.60 |
| Gaussian | FedAvg | 0.1000 | 0.8000 | 30 | ok | 18-53-21_gaussian |
| Gaussian | FLTrust | 0.4888 | 0.8000 | 30 | ok | 16-32-19_gaussian_FLtrust_cifar |
| Gaussian | FLDetector | 0.1000 | 0.8000 | 30 | ok | 14-13-22_gaussian_FLdetector_cifar |
| Gaussian | Multi-Krum | 0.2393 | 0.8000 | 30 | ok | 19-48-01_gaussian_lamda3_cifar_multikrum |
| Gaussian | Our Method | 0.6203 | 0.7000 | 100 | ok | 21-19-44_cifar10_gaussian_tau0.70 |
| LIE | FedAvg | 0.1000 | 0.8000 | 30 | ok | 19-08-43_lie |
| LIE | FLTrust | 0.4852 | 0.8000 | 30 | ok | 16-48-25_lie_FLtrust_cifar |
| LIE | FLDetector | 0.1000 | 0.8000 | 30 | ok | 13-17-12_lie_FLdetector_cifar |
| LIE | Multi-Krum | 0.2047 | 0.8000 | 30 | ok | 20-10-16_lie_cifar_multikrum |
| LIE | Our Method | 0.5691 | 0.1000 |  | summary_missing_dir | 00-59-56_cifar10_lie_tau0.10 |
| Min-Max | FedAvg | 0.1000 | 0.8000 | 30 | ok | 19-35-26_min_max |
| Min-Max | FLTrust | 0.4685 | 0.8000 | 30 | ok | 17-06-21_minmax_FLtrust_cifar |
| Min-Max | FLDetector | 0.1000 | 0.8000 | 30 | ok | 13-06-08_minmax_FLdetector_cifar |
| Min-Max | Multi-Krum | 0.2398 | 0.8000 | 30 | ok | 20-24-20_min_max_cifar_multikrum |
| Min-Max | Our Method | 0.5911 | 0.5000 |  | summary_missing_dir | 12-29-48_cifar10_min_max_tau0.50 |
| No Attack | FedAvg | 0.6170 | 0.8000 | 30 | ok | 18-41-29_no_attack |
| No Attack | FLTrust | 0.5006 | 0.8000 | 30 | ok | 16-16-02_none_FLtrust_cifar |
| No Attack | Our Method | 0.4865 | 0.0000 | 50 | ok | 02-46-35_cifar10_no_attack_tau0.00 |

## Incomplete Or Summary-Only Logs

| Status | Attack | Tau | Final Accuracy | Run | Error |
| --- | --- | --- | --- | --- | --- |
| summary_error | Gaussian | 0.1000 |  |  | Quantiles must be in the range [0, 1] |
| summary_missing_dir | Gaussian | 0.0000 | 0.5556 | 15-09-06_cifar10_gaussian_tau0.00 |  |
| summary_missing_dir | Gaussian | 0.0000 | 0.3845 | 15-48-04_cifar10_gaussian_tau0.00 |  |
| summary_error | No Attack | 0.0000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_error | No Attack | 0.1000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_error | No Attack | 0.2000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_error | No Attack | 0.3000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_error | No Attack | 0.4000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_error | No Attack | 0.5000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_error | No Attack | 0.5000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_error | No Attack | 0.6000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_error | No Attack | 0.7000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_error | No Attack | 0.8000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_error | No Attack | 0.9000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_error | No Attack | 1.0000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_missing_dir | LIE | 0.0000 | 0.1000 | 00-16-21_cifar10_lie_tau0.00 |  |
| summary_missing_dir | LIE | 0.1000 | 0.5691 | 00-59-56_cifar10_lie_tau0.10 |  |
| summary_missing_dir | LIE | 0.2000 | 0.1000 | 01-43-38_cifar10_lie_tau0.20 |  |
| summary_missing_dir | LIE | 0.3000 | 0.1000 | 02-27-11_cifar10_lie_tau0.30 |  |
| summary_missing_dir | LIE | 0.4000 | 0.1000 | 03-10-50_cifar10_lie_tau0.40 |  |
| summary_missing_dir | LIE | 0.5000 | 0.1000 | 03-54-30_cifar10_lie_tau0.50 |  |
| summary_missing_dir | LIE | 0.6000 | 0.1000 | 04-38-10_cifar10_lie_tau0.60 |  |
| summary_missing_dir | LIE | 0.7000 | 0.1000 | 05-21-56_cifar10_lie_tau0.70 |  |
| summary_missing_dir | LIE | 0.8000 | 0.1000 | 06-05-41_cifar10_lie_tau0.80 |  |
| summary_missing_dir | LIE | 0.9000 | 0.1000 | 06-49-26_cifar10_lie_tau0.90 |  |
| summary_missing_dir | LIE | 1.0000 | 0.1000 | 07-33-12_cifar10_lie_tau1.00 |  |
| summary_missing_dir | Min-Max | 0.0000 | 0.1000 | 08-17-04_cifar10_min_max_tau0.00 |  |
| summary_missing_dir | Min-Max | 0.1000 | 0.1000 | 09-07-31_cifar10_min_max_tau0.10 |  |
| summary_missing_dir | Min-Max | 0.2000 | 0.1000 | 09-58-00_cifar10_min_max_tau0.20 |  |
| summary_missing_dir | Min-Max | 0.3000 | 0.1000 | 10-48-44_cifar10_min_max_tau0.30 |  |
| summary_missing_dir | Min-Max | 0.4000 | 0.1000 | 11-39-17_cifar10_min_max_tau0.40 |  |
| summary_missing_dir | Min-Max | 0.5000 | 0.5911 | 12-29-48_cifar10_min_max_tau0.50 |  |
| missing_results_summary | LIE | 0.0000 | 0.4808 | 13-35-17_cifar10_lie_tau0.00 |  |
| missing_results_summary | LIE | 0.1000 | 0.4262 | 14-20-44_cifar10_lie_tau0.10 |  |
| missing_results_summary | LIE | 0.2000 | 0.4550 | 15-05-52_cifar10_lie_tau0.20 |  |
| missing_results_summary | LIE | 0.3000 | 0.4483 | 15-51-18_cifar10_lie_tau0.30 |  |
| missing_results | LIE | 0.4000 |  | 16-37-09_cifar10_lie_tau0.40 |  |
| summary_missing_dir | Min-Max | 0.0000 | 0.4629 | 18-26-10_cifar10_min_max_tau0.00 |  |
| summary_error | Min-Max | 0.0000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_error | Min-Max | 0.0000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_error | Min-Max | 0.1000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_error | Min-Max | 0.1000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_error | Min-Max | 0.2000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_error | Min-Max | 0.3000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| summary_error | Min-Max | 0.4000 |  |  | Message content is None. Use <message>.has_content() to check if a message has content. |
| missing_results | Min-Max | 0.1000 |  | 19-40-02_cifar10_min_max_tau0.10 |  |
