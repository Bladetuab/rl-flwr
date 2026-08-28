# Task 2 Implementation Report: Wire dataset selection into client_app.py

## Summary
Task 2 has been successfully completed. All three modification steps from the brief have been applied to `rl_flwr/client_app.py`, and the verification command passed.

## Implementation Details

### Step 1: Update the import (✓)
**File:** `rl_flwr/client_app.py:7-9`

Changed from:
```python
from rl_flwr.task import Net, load_data
```

To:
```python
from rl_flwr.task import get_net, load_data
```

**Status:** Applied successfully. The import now references `get_net` instead of the `Net` class.

### Step 2: Make `train()` dataset-aware (✓)
**File:** `rl_flwr/client_app.py:15-30`

Key changes:
- Added `dataset_name = msg.content["config"].get("dataset", "fashion_mnist")` at the beginning to extract dataset from message config with safe fallback
- Changed `model = Net()` to `model = get_net(dataset_name)` to use dataset-specific model factory
- Changed `load_data(partition_id, num_partitions)` to `load_data(partition_id, num_partitions, dataset_name=dataset_name)` to pass dataset to data loader

**Status:** Applied successfully. The train handler now reads dataset from config and uses the parametrized signatures.

### Step 3: Make `evaluate()` dataset-aware (✓)
**File:** `rl_flwr/client_app.py:68-83`

Key changes identical to Step 2:
- Added `dataset_name = msg.content["config"].get("dataset", "fashion_mnist")` 
- Changed `model = Net()` to `model = get_net(dataset_name)`
- Changed `load_data(partition_id, num_partitions)` to `load_data(partition_id, num_partitions, dataset_name=dataset_name)`

**Status:** Applied successfully. The evaluate handler now has matching dataset awareness.

### Step 4: Verification (✓)
**Command run:**
```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && python -c "import rl_flwr.client_app; print('ok')"
```

**Actual output:**
```
ok
```

**Status:** Import check passed. The module loads without errors.

## Backward Compatibility

The use of `.get("dataset", "fashion_mnist")` instead of direct dictionary access ensures backward compatibility with `server_app.py` (which is out of scope and untouched). When `server_app.py` does not include a `dataset` key in the config, the default value "fashion_mnist" is used, maintaining existing behavior.

## Files Changed
- `/home/tuan/get-local-model-bt-id/rl-flwr/rl_flwr/client_app.py` (3 edits: import line, train function, evaluate function)

## Self-Review Findings
All self-review checklist items passed:
- ✅ All steps applied in correct order
- ✅ Import uses `get_net, load_data` (not `Net`)
- ✅ Both functions use `.get("dataset", "fashion_mnist")` with safe default
- ✅ Step 4 verification command printed `ok`

## Notes
- No issues or concerns identified
- The implementation exactly matches the brief specifications
- Interfaces with Task 1 functions (`get_net` and `load_data`) as designed
- Ready for Task 3 and subsequent tasks
