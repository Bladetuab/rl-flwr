import math
from typing import Dict, List, Tuple

import numpy as np
import torch
from flwr.app import Message, ArrayRecord
from flwr.common import log
from logging import INFO
from scipy.stats import norm


def no_attack_messages(
    replies: List[Message],
    clients_state: Dict[int, bool],
    **kwargs,
) -> List[Message]:
    """No-op attack: trả về nguyên xi."""
    return replies


def gaussian_attack_messages(
    replies: List[Message],
    clients_state: Dict[int, bool],
    **kwargs,
) -> List[Message]:
    """Gaussian attack trên tham số model trong ArrayRecord.

    Parameters
    ----------
    replies:
        Danh sách Message từ client (sau train), như `train_replies` của strategy.
    clients_state:
        Dict[node_id -> bool], True nếu client là malicious.
        (node_id = msg.metadata.src_node_id)
    magnitude: float
        Độ lệch chuẩn của nhiễu Gaussian (sigma).
    dataset_name: str
        Optional: giữ tham số này nếu bạn muốn xử lý khác nhau.

    Returns
    -------
    attacked_replies:
        Danh sách Message sau khi bị/không bị attack (cùng chiều với `replies`).
    """

    magnitude = float(kwargs.get("magnitude", 0.0))
    dataset_name = kwargs.get("dataset_name", "no name")

    if magnitude <= 0.0:
        # Không attack nếu magnitude = 0
        return replies

    attacked_replies: List[Message] = []

    def perturb_numpy(arr: np.ndarray) -> np.ndarray:
        """Thêm nhiễu Gaussian lên numpy array."""
        return arr + np.random.normal(loc=0.0, scale=magnitude, size=arr.shape)

    for msg in replies:
        # Node id của client gửi message này
        node_id = msg.metadata.src_node_id
        is_malicious = clients_state.get(node_id, False)

        if not is_malicious:
            # Giữ nguyên message nếu không phải malicious
            attacked_replies.append(msg)
            continue

        # Malicious client: thêm Gaussian noise lên weights
        # 1) Lấy ArrayRecord chứa model weights
        arrays: ArrayRecord = msg.content["arrays"]

        # 2) Convert sang torch state_dict cho dễ thao tác (chuẩn docs)
        #    https://flower.ai/docs/framework/ref-api/flwr.app.ArrayRecord.html#flwr.app.ArrayRecord.to_torch_state_dict :contentReference[oaicite:0]{index=0}
        state_dict = arrays.to_torch_state_dict()

        # 3) Thêm nhiễu lên từng tensor trong state_dict
        for name, tensor in state_dict.items():
            # tensor: torch.Tensor
            noise = torch.normal(
                mean=0.0,
                std=magnitude,
                size=tensor.shape,
                device=tensor.device,
            )
            state_dict[name] = tensor + noise

        # 4) Convert ngược lại thành ArrayRecord
        #    https://flower.ai/docs/framework/ref-api/flwr.app.ArrayRecord.html#flwr.app.ArrayRecord.from_torch_state_dict :contentReference[oaicite:1]{index=1}
        attacked_arrays = ArrayRecord.from_torch_state_dict(state_dict)

        # 5) Gán lại vào Message (in-place)
        msg.content["arrays"] = attacked_arrays

        log(
            INFO,
            "Applied Gaussian attack to client node_id=%s with magnitude=%s",
            node_id,
            magnitude,
        )

        attacked_replies.append(msg)

    return attacked_replies



# ==========================================================================
#  Attack enhancement: de-correlating the shared payload
# ==========================================================================
def _decorrelate_payload(
    payload: Dict[str, np.ndarray],
    target_cos: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Return one client's copy of ``payload`` perturbed so that independent
    draws have an expected pairwise cosine of ``target_cos``.

    Why this exists
    ---------------
    As published, LIE and Min-Max make every malicious client submit the
    *identical* vector. Any defense with a pairwise-similarity or clustering
    signal (USTAD-FL's clone score, SignGuard's sign clustering, DnC's top
    singular direction) then separates the attackers trivially, and a perfect
    detection score measures the implementation rather than the detector.

    Scale derivation
    ----------------
    For payload p and independent noise e_i ~ N(0, s^2 I) in d dimensions,
    with d large,  E[cos(p+e_i, p+e_j)] ~= ||p||^2 / (||p||^2 + s^2 d).
    Setting that equal to ``target_cos`` gives

        s = ||p|| * sqrt( (1 - target_cos) / (target_cos * d) )

    so the caller specifies the *detectability* they want rather than an
    arbitrary noise std. The achieved cosine is measured and logged by the
    caller, not assumed.
    """
    flat = np.concatenate([payload[k].ravel() for k in payload])
    d = flat.size
    p_norm = float(np.linalg.norm(flat))
    if d == 0 or p_norm == 0.0 or not (0.0 < target_cos < 1.0):
        return {k: v.copy() for k, v in payload.items()}

    s = p_norm * math.sqrt((1.0 - target_cos) / (target_cos * d))
    return {
        k: (v + rng.normal(0.0, s, size=v.shape)).astype(v.dtype)
        for k, v in payload.items()
    }


def _max_pairwise_cosine(vecs: List[np.ndarray]) -> float:
    """Max pairwise cosine over a list of flattened vectors (diagnostic)."""
    if len(vecs) < 2:
        return float("nan")
    M = np.vstack([v / (np.linalg.norm(v) + 1e-12) for v in vecs])
    G = M @ M.T
    np.fill_diagonal(G, -np.inf)
    return float(np.max(G))


def lie_attack_messages(
    replies: List[Message],
    clients_state: Dict[int, bool],
    omniscent: bool = True,
    **kwargs,
) -> List[Message]:
    """Apply Omniscient LIE attack (Baruch et al., 2019) on parameters.

    Parameters
    ----------
    replies
        List of Messages from clients (after train), như train_replies.
    clients_state
        Dict[node_id -> bool], True nếu client là malicious.
    omniscent
        Giữ để tương thích, nhưng logic giống code gốc (cờ này gần như không dùng).

    Returns
    -------
    attacked_replies
        List[Message] sau khi áp dụng LIE lên các client malicious.
    """

    # Sao chép list để không sửa trực tiếp đối số
    attacked_replies: List[Message] = list(replies)

    if not attacked_replies:
        return attacked_replies

    # ------------------------------------------------------------------
    # 1) Lấy state_dict của từng client, đảm bảo cùng bộ key
    # ------------------------------------------------------------------
    state_dicts: List[Dict[str, torch.Tensor]] = []
    node_ids: List[int] = []

    for msg in attacked_replies:
        node_id = msg.metadata.src_node_id
        node_ids.append(node_id)

        arrays: ArrayRecord = msg.content["arrays"]
        sd = arrays.to_torch_state_dict()  # giữ nguyên tên tham số
        state_dicts.append(sd)

    # Lấy danh sách key từ client đầu tiên và giả sử tất cả giống nhau
    param_keys = list(state_dicts[0].keys())

    # ------------------------------------------------------------------
    # 2) Tính grads_mean và grads_stdev cho từng tham số (theo key)
    # ------------------------------------------------------------------
    grads_mean: Dict[str, np.ndarray] = {}
    grads_stdev: Dict[str, np.ndarray] = {}

    for key in param_keys:
        # stack tất cả tham số cùng key từ mọi client: shape (num_clients, *shape)
        tensors = [sd[key].detach().cpu().numpy() for sd in state_dicts]
        stacked = np.stack(tensors, axis=0)

        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)

        # Giữ behavior giống code gốc: std ** 0.5 (dù hơi kỳ)
        grads_mean[key] = mean
        grads_stdev[key] = std ** 0.5

    # (Giống code gốc: cờ omniscent không thực sự thay đổi mean/std,
    # nên ta giữ nguyên behavior.)

    num_clients = len(attacked_replies)
    num_malicious = sum(
        1 for nid in set(node_ids) if clients_state.get(nid, False)
    )

    if num_malicious == 0:
        # Không có client malicious -> không làm gì
        return attacked_replies

    # ------------------------------------------------------------------
    # 3) Tính z_max theo công thức LIE
    # ------------------------------------------------------------------
    num_supporters = math.floor((num_clients / 2) + 1) - num_malicious

    z_max = norm.cdf(
        (num_clients - num_malicious - num_supporters) / (num_clients - num_malicious)
    )

    # ------------------------------------------------------------------
    # 4) Gán tham số LIE cho từng client malicious, GIỮ NGUYÊN KEY
    # ------------------------------------------------------------------
    # Shared LIE payload: mean - z_max * std, identical for every attacker.
    base_payload = {
        key: grads_mean[key] - z_max * grads_stdev[key] for key in param_keys
    }
    decorr_cos = kwargs.get("decorrelate_cos", None)
    _rng = np.random.default_rng(int(kwargs.get("decorrelate_seed", 1234)))
    _sent: List[np.ndarray] = []

    for idx, msg in enumerate(attacked_replies):
        node_id = msg.metadata.src_node_id
        if not clients_state.get(node_id, False):
            continue  # chỉ sửa client malicious

        sd = state_dicts[idx]

        payload = (
            _decorrelate_payload(base_payload, float(decorr_cos), _rng)
            if decorr_cos else base_payload
        )
        _sent.append(np.concatenate([payload[k].ravel() for k in param_keys]))

        for key in param_keys:
            new_np = payload[key]

            # chuyển np -> tensor trên đúng device/dtype
            old_tensor = sd[key]
            new_tensor = torch.from_numpy(new_np).to(
                device=old_tensor.device, dtype=old_tensor.dtype
            )
            sd[key] = new_tensor

        # Tạo lại ArrayRecord từ state_dict (key giữ nguyên)
        attacked_arrays = ArrayRecord.from_torch_state_dict(sd)
        msg.content["arrays"] = attacked_arrays

        log(
            INFO,
            "Applied LIE attack to client node_id=%s (z_max=%.4f)",
            node_id,
            z_max,
        )

    if _sent:
        log(INFO, "[LIE] payload de-correlation: n_mal=%d  max_pairwise_cos=%.4f  (target=%s)",
            len(_sent), _max_pairwise_cosine(_sent), str(decorr_cos))

    return attacked_replies



def _compute_distances_state_dicts(
    weights: List[Dict[str, np.ndarray]],
) -> np.ndarray:
    """Compute squared distances between flattened weight vectors.

    weights: list of dict[param_name -> np.ndarray]
    """
    flat_w = [
        np.concatenate([par.ravel() for par in w.values()], axis=0)
        for w in weights
    ]
    flat_w = np.stack(flat_w, axis=0)  # shape: (n_clients, dim)

    n = flat_w.shape[0]
    distance_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            delta = flat_w[i] - flat_w[j]
            dist = np.linalg.norm(delta)
            distance_matrix[i, j] = dist**2

    return distance_matrix


def minmax_attack_messages(
    replies: List[Message],
    clients_state: Dict[int, bool],
    omniscent: bool = True,
    **kwargs,
) -> List[Message]:
    """Apply Min-Max agnostic attack on Messages (Message API version).

    Based on:
    "Manipulating the Byzantine: Optimizing Model Poisoning Attacks and
    Defenses for Federated Learning" (Shejwalkar et al., 2021).

    Parameters
    ----------
    replies
        List[Message] from clients after train (e.g. `train_replies`).
    clients_state
        Dict[node_id -> bool], True if malicious, False otherwise.
    omniscent
        Kept for compatibility with original code.
        (Hành vi hiện tại tương đương omniscient attacker.)
    threshold : float (kwarg)
        Threshold for lambda convergence.
    lambda_init : float (kwarg, key: "lambda")
        Initial value for lambda.
    malicious_num : int (kwarg)
        Number of malicious clients.
    dataset_name : str (kwarg)
        Dataset name ("mnist", "cifar", ...) – dùng để chọn perturbation.

    Returns
    -------
    attacked_replies
        List[Message] sau khi áp dụng Min-Max attack lên client malicious.
    """

    attacked_replies: List[Message] = list(replies)

    if not attacked_replies:
        return attacked_replies

    dataset_name = kwargs.get("dataset_name", None)
    threshold = float(kwargs.get("threshold", 1e-5))
    lambda_init = float(kwargs.get("lambda", 5.0))
    malicious_num = int(kwargs.get("malicious_num", 0))

    # ------------------------------------------------------------------
    # 1) Lấy state_dict của từng client, đảm bảo cùng bộ key
    # ------------------------------------------------------------------
    state_dicts: List[Dict[str, torch.Tensor]] = []
    node_ids: List[int] = []

    for msg in attacked_replies:
        node_id = msg.metadata.src_node_id
        node_ids.append(node_id)

        arrays: ArrayRecord = msg.content["arrays"]
        # chuyển ArrayRecord -> PyTorch state_dict (theo docs) :contentReference[oaicite:2]{index=2}
        sd = arrays.to_torch_state_dict()
        state_dicts.append(sd)

    param_keys = list(state_dicts[0].keys())

    # ------------------------------------------------------------------
    # 2) Chuyển sang numpy cho việc tính toán (per key)
    # ------------------------------------------------------------------
    # params_per_client: list[Dict[str, np.ndarray]]
    params_per_client: List[Dict[str, np.ndarray]] = []
    for sd in state_dicts:
        params_np = {k: v.detach().cpu().numpy() for k, v in sd.items()}
        params_per_client.append(params_np)

    # params_avg[key] = mean over clients
    params_avg: Dict[str, np.ndarray] = {}
    for key in param_keys:
        all_layers = [p[key] for p in params_per_client]
        stacked = np.stack(all_layers, axis=0)
        params_avg[key] = np.mean(stacked, axis=0)

    # ------------------------------------------------------------------
    # 3) Chọn perturbation_vect theo dataset_name (như code gốc)
    # ------------------------------------------------------------------
    perturbation_vect: Dict[str, np.ndarray] = {}

    for key in param_keys:
        all_layers = [p[key] for p in params_per_client]
        stacked = np.stack(all_layers, axis=0)
        std = np.std(stacked, axis=0)

        # Trong code gốc, mnist/cifar đều dùng -std
        if dataset_name in ("mnist", "cifar"):
            perturbation_vect[key] = -std
        else:
            perturbation_vect[key] = -std

    num_clients = len(attacked_replies)

    # Đếm lại malicious_num nếu không truyền từ ngoài
    if malicious_num <= 0:
        malicious_num = sum(
            1 for nid in set(node_ids) if clients_state.get(nid, False)
        )

    if malicious_num == 0:
        # Không có client malicious -> không làm gì
        return attacked_replies

    # ------------------------------------------------------------------
    # 4) Tìm lambda tối ưu (gamma trong paper) bằng binary search thô
    # ------------------------------------------------------------------
    lambda_succ = lambda_init + 1.0
    curr_lambda = lambda_init
    step = lambda_init * 0.5

    # indices của malicious và benign
    malicious_indices = [
        i for i, nid in enumerate(node_ids) if clients_state.get(nid, False)
    ]
    benign_indices = [
        i for i, nid in enumerate(node_ids) if not clients_state.get(nid, False)
    ]

    while (
        abs(lambda_succ - curr_lambda) > threshold
        and step > threshold
        and malicious_num > 0
    ):
        # Compute malicious gradients/updates (corrupted_params) theo curr_lambda
        perturbed_params: Dict[str, np.ndarray] = {
            key: curr_lambda * perturbation_vect[key] for key in param_keys
        }
        corrupted_params: Dict[str, np.ndarray] = {
            key: params_avg[key] + perturbed_params[key] for key in param_keys
        }

        # Build params_c: weights của từng client với malicious bị thay bằng corrupted_params
        params_c: List[Dict[str, np.ndarray]] = []
        for idx, params_np in enumerate(params_per_client):
            nid = node_ids[idx]
            if clients_state.get(nid, False):
                # malicious -> dùng corrupted_params
                params_c.append(corrupted_params)
            else:
                params_c.append(params_np)

        distance_matrix = _compute_distances_state_dicts(params_c)

        # Benign-benign distances
        if len(benign_indices) > 1:
            dm_b = distance_matrix[
                np.ix_(benign_indices, benign_indices)
            ]
            max_dist_b = np.max(dm_b)
        else:
            # nếu <2 benign, đặt max_dist_b lớn để tránh chia lỗi
            max_dist_b = float("inf")

        # Malicious-benign distances
        if len(malicious_indices) > 0 and len(benign_indices) > 0:
            dm_m = distance_matrix[
                np.ix_(malicious_indices, benign_indices)
            ]
            max_dist_m = np.max(dm_m)
        else:
            max_dist_m = 0.0

        # Update lambda theo điều kiện Min-Max
        if max_dist_m < max_dist_b:
            # Lambda (gamma) tốt, lưu lại và thử tăng
            lambda_succ = curr_lambda
            curr_lambda = curr_lambda + step * 0.5
        else:
            # Lambda quá lớn, giảm bớt
            curr_lambda = curr_lambda - step * 0.5

        step *= 0.5

    # ------------------------------------------------------------------
    # 5) Tính final malicious update và ghi vào các client malicious
    # ------------------------------------------------------------------
    final_perturb: Dict[str, np.ndarray] = {
        key: lambda_succ * perturbation_vect[key] for key in param_keys
    }
    final_corrupted: Dict[str, np.ndarray] = {
        key: params_avg[key] + final_perturb[key] for key in param_keys
    }

    decorr_cos = kwargs.get("decorrelate_cos", None)
    _rng = np.random.default_rng(int(kwargs.get("decorrelate_seed", 1234)))
    _sent: List[np.ndarray] = []

    for idx, msg in enumerate(attacked_replies):
        nid = msg.metadata.src_node_id
        if not clients_state.get(nid, False):
            continue  # chỉ sửa client malicious

        payload = (
            _decorrelate_payload(final_corrupted, float(decorr_cos), _rng)
            if decorr_cos else final_corrupted
        )
        _sent.append(np.concatenate([payload[k].ravel() for k in param_keys]))

        sd = state_dicts[idx]
        for key in param_keys:
            old_tensor = sd[key]
            new_np = payload[key]
            new_tensor = torch.from_numpy(new_np).to(
                device=old_tensor.device, dtype=old_tensor.dtype
            )
            sd[key] = new_tensor

        # Convert back to ArrayRecord with same keys (rất quan trọng) :contentReference[oaicite:3]{index=3}
        attacked_arrays = ArrayRecord.from_torch_state_dict(sd)
        msg.content["arrays"] = attacked_arrays

        log(
            INFO,
            "Applied Min-Max attack to client node_id=%s (lambda=%.6f)",
            nid,
            lambda_succ,
        )

    if _sent:
        log(INFO, "[Min-Max] payload de-correlation: n_mal=%d  max_pairwise_cos=%.4f  (target=%s)",
            len(_sent), _max_pairwise_cosine(_sent), str(decorr_cos))

    return attacked_replies





def _compute_distances_state_dicts(
    weights: List[Dict[str, np.ndarray]],
) -> np.ndarray:
    """Compute squared distances between flattened weight vectors.

    Parameters
    ----------
    weights : list of dict[param_name -> np.ndarray]

    Returns
    -------
    distance_matrix : np.ndarray of shape (n_clients, n_clients)
    """
    flat_w = [
        np.concatenate([par.ravel() for par in w.values()], axis=0) for w in weights
    ]
    flat_w = np.stack(flat_w, axis=0)  # (n_clients, dim)

    n = flat_w.shape[0]
    distance_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            delta = flat_w[i] - flat_w[j]
            dist = np.linalg.norm(delta)
            distance_matrix[i, j] = dist**2

    return distance_matrix


def _get_closest_indices(distance_matrix: np.ndarray, num_closest: int) -> List[List[int]]:
    """Get indices of the closest points for each vector."""
    closest_indices: List[List[int]] = []
    for idx, _ in enumerate(distance_matrix):
        # bỏ index 0 (chính nó), lấy num_closest điểm gần nhất
        closest = np.argsort(distance_matrix[idx])[1 : num_closest + 1].tolist()
        closest_indices.append(closest)
    return closest_indices


def _krum_state_dicts(
    results: List[Tuple[Dict[str, np.ndarray], int]],
    num_malicious: int,
    to_keep: int,
    num_closest: int | None = None,
) -> Tuple[Dict[str, np.ndarray], int, np.ndarray, List[float]]:
    """Krum trên list state_dict (numpy). Giống logic _krum gốc.

    Parameters
    ----------
    results : list of (weights_dict, num_examples)
    num_malicious : int
    to_keep : int
    num_closest : int or None

    Returns
    -------
    best_weights : dict[param_name -> np.ndarray]
    best_index : int
    best_indices : np.ndarray
    scores : list[float]
    """
    weights = [w for (w, _) in results]
    distance_matrix = _compute_distances_state_dicts(weights)

    if not num_closest:
        num_closest = len(weights) - num_malicious - 2
    if num_closest <= 0:
        num_closest = 1
    elif num_closest > len(weights):
        num_closest = len(weights)

    closest_indices = _get_closest_indices(distance_matrix, num_closest)

    scores = [
        float(np.sum(distance_matrix[i, closest_indices[i]]))
        for i in range(len(distance_matrix))
    ]

    best_index = int(np.argmin(scores))
    best_indices = np.argsort(scores)[::-1][len(scores) - to_keep :]

    return weights[best_index], best_index, best_indices, scores


def _fang_corrupt_params_state_dict(
    global_model_np: Dict[str, np.ndarray], lamda: float
) -> Dict[str, np.ndarray]:
    """Tạo corrupted_params = global_model - sign(global_model) * lambda (per param)."""
    corrupted: Dict[str, np.ndarray] = {}
    for k, v in global_model_np.items():
        magnitude = np.sign(v) * lamda
        corrupted[k] = v - magnitude
    return corrupted


def _fang_corrupt_and_select_state_dict(
    all_models: List[Tuple[Dict[str, np.ndarray], int]],
    global_model_np: Dict[str, np.ndarray],
    node_ids: List[int],
    clients_state: Dict[int, bool],
    num_corrupted: int,
    lamda: float,
) -> Tuple[bool, Dict[str, np.ndarray]]:
    """Corrupt malicious models với lambda và check xem Krum chọn malicious không."""
    corrupted_params = _fang_corrupt_params_state_dict(global_model_np, lamda)

    # Thay model của client malicious bằng corrupted_params
    all_models_m: List[Tuple[Dict[str, np.ndarray], int]] = []
    for idx, (model_np, num_ex) in enumerate(all_models):
        nid = node_ids[idx]
        if clients_state.get(nid, False):
            all_models_m.append((corrupted_params, num_ex))
        else:
            all_models_m.append((model_np, num_ex))

    _, best_index, _, _ = _krum_state_dicts(all_models_m, num_corrupted, 1)

    # Kiểm tra best model có phải malicious không
    best_node_id = node_ids[best_index]
    malicious_selected = clients_state.get(best_node_id, False)
    return malicious_selected, corrupted_params


def fang_attack_messages(
    replies: List[Message],
    clients_state: Dict[int, bool],
    omniscent: bool = True,
    **kwargs,
) -> List[Message]:
    """Fang local model poisoning attack (Message API version).

    (Fang et al., 2020) - thiết kế cho Krum, nhưng có thể dùng cho aggregator khác.

    Parameters
    ----------
    replies
        List[Message] (train replies) từ client.
    clients_state
        Dict[node_id -> bool], True nếu client là malicious.
    omniscent
        Kept only for API compatibility. Implementation ở đây giả định attacker
        biết hết local models (omniscent=True).
    num_layers : int (kwarg, optional)
        Number of layers. Nếu không truyền, sẽ dùng len(state_dict).
    w_re : Dict[str, np.ndarray] (kwarg)  (BẮT BUỘC)
        Global model hiện tại dưới dạng dict[param_name -> np.ndarray].
        (Bạn tạo từ ArrayRecord.to_torch_state_dict + .numpy trong strategy.)
    threshold : float (kwarg)
        Threshold cho lambda.

    Returns
    -------
    attacked_replies
        List[Message] sau khi chỉnh sửa weights client malicious.
    """

    attacked_replies: List[Message] = list(replies)

    if not attacked_replies:
        return attacked_replies

    # Lấy global model w_re (numpy dict)
    w_re_np: Dict[str, np.ndarray] | None = kwargs.get("w_re", None)
    if w_re_np is None:
        raise ValueError(
            "fang_attack_messages requires 'w_re' (global model as Dict[str, np.ndarray]) in kwargs"
        )

    threshold = float(kwargs.get("threshold", 1e-5))

    # ------------------------------------------------------------------
    # 1) Lấy state_dict và convert sang numpy cho từng client
    # ------------------------------------------------------------------
    state_dicts: List[Dict[str, torch.Tensor]] = []
    params_per_client: List[Dict[str, np.ndarray]] = []
    node_ids: List[int] = []
    num_examples_list: List[int] = []

    for msg in attacked_replies:
        nid = msg.metadata.src_node_id
        node_ids.append(nid)

        arrays: ArrayRecord = msg.content["arrays"]
        sd = arrays.to_torch_state_dict()
        state_dicts.append(sd)

        params_np = {k: v.detach().cpu().numpy() for k, v in sd.items()}
        params_per_client.append(params_np)

        metrics = msg.content.get("metrics", None)
        if metrics is not None:
            num_ex = int(metrics.get("num_examples", 1))
        else:
            num_ex = 1
        num_examples_list.append(num_ex)

    param_keys = list(state_dicts[0].keys())
    num_layers = kwargs.get("num_layers", len(param_keys))

    num_clients = len(attacked_replies)
    num_corrupted = sum(1 for nid in set(node_ids) if clients_state.get(nid, False))
    # need at least 2 malicious to avoid division by 0 (giống code gốc)
    num_corrupted = max(num_corrupted, 2)

    if num_corrupted == 0:
        # Không có client malicious -> không làm gì
        return attacked_replies

    # all_params: giống all_models trong code gốc
    all_params: List[Tuple[Dict[str, np.ndarray], int]] = list(
        zip(params_per_client, num_examples_list)
    )

    # benign: list các client benign (model, num_examples)
    benign: List[Tuple[Dict[str, np.ndarray], int]] = [
        (params_per_client[i], num_examples_list[i])
        for i, nid in enumerate(node_ids)
        if not clients_state.get(nid, False)
    ]

    # ------------------------------------------------------------------
    # 2) Tính "scores" (gọi nhầm là distances trong code gốc) của Krum
    # ------------------------------------------------------------------
    _, _, _, scores = _krum_state_dicts(all_params, num_corrupted, 1)

    # index của benign clients trong all_params
    idx_benign = [
        i for i, nid in enumerate(node_ids) if not clients_state.get(nid, False)
    ]
    if len(idx_benign) == 0:
        # Không có benign -> không tối ưu được, coi như bỏ qua attack
        return attacked_replies

    min_dist = float(np.min(np.array(scores)[idx_benign])) / (
        ((num_clients - 2) * (num_corrupted - 1)) * math.sqrt(num_layers)
    )

    # ------------------------------------------------------------------
    # 3) Tính max distance từ w_re (global) đến từng benign
    # ------------------------------------------------------------------
    dist_wre = np.zeros((len(benign),), dtype=np.float64)
    for i, (b_params, _) in enumerate(benign):
        # diff per layer: b_params[key] - w_re_np[key]
        norm_sums = 0.0
        for key in param_keys:
            diff = b_params[key] - w_re_np[key]
            norm_sums += np.linalg.norm(diff)
        dist_wre[i] = norm_sums**2

    max_dist = float(np.max(dist_wre)) / math.sqrt(num_layers)

    lamda = min(min_dist + max_dist, 999.0)

    # ------------------------------------------------------------------
    # 4) Lặp điều chỉnh lambda cho tới khi Krum chọn malicious
    # ------------------------------------------------------------------
    malicious_selected, corrupted_params_np = _fang_corrupt_and_select_state_dict(
        all_params,
        w_re_np,
        node_ids,
        clients_state,
        num_corrupted,
        lamda,
    )

    while lamda > threshold and not malicious_selected:
        lamda = lamda * 0.5
        malicious_selected, corrupted_params_np = _fang_corrupt_and_select_state_dict(
            all_params,
            w_re_np,
            node_ids,
            clients_state,
            num_corrupted,
            lamda,
        )   

    log(
        INFO,
        "Fang attack finished with lambda=%.6f, malicious_selected=%s",
        lamda,
        malicious_selected,
    )

    # ------------------------------------------------------------------
    # 5) Gán corrupted_params cho tất cả client malicious, GIỮ NGUYÊN KEY
    # ------------------------------------------------------------------
    for idx, msg in enumerate(attacked_replies):
        nid = msg.metadata.src_node_id
        if not clients_state.get(nid, False):
            continue  # chỉ sửa client malicious

        sd = state_dicts[idx]
        for key in param_keys:
            old_tensor = sd[key]
            new_np = corrupted_params_np[key]
            new_tensor = torch.from_numpy(new_np).to(
                device=old_tensor.device,
                dtype=old_tensor.dtype,
            )
            sd[key] = new_tensor

        attacked_arrays = ArrayRecord.from_torch_state_dict(sd)
        msg.content["arrays"] = attacked_arrays

        log(
            INFO,
            "Applied Fang attack to client node_id=%s (lambda=%.6f)",
            nid,
            lamda,
        )

    return attacked_replies



