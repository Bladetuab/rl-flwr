# Task 1: Dataset/Model Registry Implementation Report

## Summary
Successfully implemented all four steps of the dataset/model registry in `rl_flwr/task.py`. The task adds support for multiple datasets (MNIST, Fashion-MNIST, CIFAR-10) through a centralized registry and model selection system.

## Implementation Steps

### Step 1: Activate NetCIFAR Class
- **File:** `rl_flwr/task.py` lines 33-51
- **Change:** Replaced commented-out CIFAR-10 configuration block with an active `NetCIFAR` class
- **Details:** The class is a 3-channel CNN adapted for 32×32 CIFAR-10 images with correct flatten dimensions (16 × 5 × 5 = 400)

### Step 2: Add Dataset Registry and Helper Functions
- **File:** `rl_flwr/task.py` lines 66-113
- **Functions added:**
  - `DATASET_REGISTRY`: Dictionary mapping dataset names to configuration (HuggingFace path, image key, channels, size, normalization)
  - `get_net(dataset_name: str) -> nn.Module`: Returns appropriate model for dataset
  - `build_transforms(dataset_name: str)`: Returns (train_transform, eval_transform) tuple with dataset-specific normalization and augmentation

### Step 3: Shape Sanity Check
- **Command run:**
  ```bash
  python -c "
  import torch
  from rl_flwr.task import get_net, DATASET_REGISTRY
  
  for name, cfg in DATASET_REGISTRY.items():
      net = get_net(name)
      x = torch.randn(2, cfg['channels'], cfg['image_size'], cfg['image_size'])
      out = net(x)
      print(name, '->', tuple(out.shape))
  "
  ```
- **Actual output:**
  ```
  mnist -> (2, 10)
  fashion_mnist -> (2, 10)
  cifar10 -> (2, 10)
  ```
- **Status:** PASS (All three datasets produce correct (2, 10) output shape)

### Step 4: Add Transform-Closure Factories and Rewrite load_data
- **File:** `rl_flwr/task.py` lines 188-240
- **Functions added:**
  - `make_train_transform_fn(transform)`: Creates batch-compatible wrapper for train transforms
  - `make_eval_transform_fn(transform)`: Creates batch-compatible wrapper for eval transforms
  - `_fds_cache`: Dict to cache FederatedDataset instances per dataset
  - Rewrote `load_data()` to accept `dataset_name` parameter (default: "fashion_mnist") and handle multiple datasets with appropriate transforms and column renaming (e.g., "img" → "image" for CIFAR-10)

### Step 5: Import Verification
- **Command run:**
  ```bash
  python -c "import rl_flwr.task; print('ok')"
  ```
- **Actual output:**
  ```
  ok
  ```
- **Status:** PASS (No SyntaxError, ImportError, or NameError)

## Self-Review Checklist

- [x] All steps in the brief applied in order
- [x] `build_fltrust_rootset_loader` still present and untouched
- [x] Original `fds` global variable still present and untouched
- [x] `apply_train_transforms` and `apply_eval_transforms` still present and untouched (used by `build_fltrust_rootset_loader`)
- [x] `FM_NORMALIZATION`, `EVAL_TRANSFORMS`, `TRAIN_TRANSFORMS` still present and untouched
- [x] Original `Net` class still present and untouched
- [x] Step 3 shape-check script printed all 3 expected lines (mnist → (2, 10), fashion_mnist → (2, 10), cifar10 → (2, 10))
- [x] Step 5 import check printed "ok"
- [x] All interface names match brief exactly:
  - [x] `get_net` (function)
  - [x] `build_transforms` (function)
  - [x] `make_train_transform_fn` (function)
  - [x] `make_eval_transform_fn` (function)
  - [x] `DATASET_REGISTRY` (dict)
  - [x] `NetCIFAR` (class)
  - [x] `load_data` (function with dataset_name parameter)

## Files Changed

- `/home/tuan/get-local-model-bt-id/rl-flwr/rl_flwr/task.py`: Modified with 4 implementation steps

## Verification Results

| Step | Command | Result |
|------|---------|--------|
| 3 | Shape check | PASS - All 3 datasets output (2, 10) |
| 5 | Import check | PASS - "ok" printed |

## Notes

- The registry design supports arbitrary future datasets via simple dict entries
- Transform factories are properly isolated and safe for use with HuggingFace Dataset.with_transform()
- The `_fds_cache` dict allows Tasks 2-3 to work with multiple datasets without re-loading from HuggingFace repeatedly
- Backward compatibility maintained: `load_data()` defaults to "fashion_mnist" if no dataset specified
- All original functions used by `build_fltrust_rootset_loader` remain completely unchanged
