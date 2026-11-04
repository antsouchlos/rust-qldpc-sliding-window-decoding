# %%


import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import stim
import matplotlib.pyplot as plt

from quits import ErrorModel, CircuitBuildOptions
from quits.qldpc_code import BbCode
from quits.simulation import get_stim_mem_result
from quits.decoder import detector_error_model_to_matrix, sliding_window_circuit_mem

from ldpc.bp_decoder import BpDecoder as LdpcDecoder
from rust_qldpc import (
    SlidingWindowMinSumDecoder,
    SlidingWindowSpaDecoder,
)

from tqdm import tqdm


# %% [markdown]

"""
## Helper functions
"""

# %%


def build_bb_72_circuit(p: float, num_rounds: int):
    code = BbCode(
        l=6,
        m=6,
        A_x_pows=[3],
        A_y_pows=[1, 2],
        B_x_pows=[1, 2],
        B_y_pows=[3],
    )

    circuit = code.build_circuit(
        error_model=ErrorModel(p, p, p, p),
        num_rounds=num_rounds,
        basis="Z",
        circuit_build_options=CircuitBuildOptions(),
        seed=1,
    )

    return code, circuit


# %% [markdown]

"""
## Compare error rates
"""

# %%


def apply_windowing_and_simulate_LER(
    code: BbCode,
    circuit: stim.Circuit,
    Decoder: type,
    decoder_params: dict,
    W: int,
    F: int,
    num_trials: int,
    seed: int = 1,
):
    detection_events, observable_flips = get_stim_mem_result(
        circuit, num_trials, seed=seed + 1
    )

    logical_pred = sliding_window_circuit_mem(
        detection_events,
        circuit,
        code.hz,
        code.lz,
        W,
        F,
        Decoder,
        Decoder,
        decoder_params,
        decoder_params,
        "channel_probs",
        "channel_probs",
        "decode",
        "decode",
        tqdm_on=True,
    )

    LER = np.mean((observable_flips - logical_pred).any(axis=1))

    return LER


def simulate_LER_batch(
    circuit: stim.Circuit,
    decoder,
    observable_matrix: csc_matrix,
    num_trials: int,
    seed: int,
):
    detection_events, observable_flips = get_stim_mem_result(
        circuit, num_trials, seed=seed
    )

    syndromes = np.asarray(detection_events, dtype=np.uint8)
    e_hats = decoder.decode_batch(syndromes)  # (num_trials, num_vns)

    logical_pred = (observable_matrix @ e_hats.T % 2).T  # (num_trials, num_logicals)
    LER = np.mean(((observable_flips - logical_pred) % 2).any(axis=1))

    return LER


# %%


seed = 1
num_rounds = 12
num_trials = 1000
max_iter = 32
W = 5
F = 1

ps = [0.001, 0.002, 0.003, 0.004, 0.005]

LERs_own_spa_forward = []
LERs_own_spa_backward = []
LERs_own_min_sum_forward = []
LERs_own_min_sum_backward = []
LERs_roffe_spa = []
LERs_roffe_min_sum = []

print("SPA (own impl):")
for p in ps:
    code, circuit = build_bb_72_circuit(p, num_rounds)

    model = circuit.detector_error_model(decompose_errors=False)
    check_matrix, observable_matrix, priors = detector_error_model_to_matrix(model)

    H_csr = csr_matrix(check_matrix)
    for direction, reverse_sliding_direction, trial_seed, results in (
        ("forward", False, seed, LERs_own_spa_forward),
        ("backward", True, seed + 2, LERs_own_spa_backward),
    ):
        decoder = SlidingWindowSpaDecoder(
            H=H_csr,
            m=code.hz.shape[0],
            num_rounds=num_rounds,
            W=W,
            F=F,
            priors=priors,
            max_iter=max_iter,
            warm_start=False,
            reverse_sliding_direction=reverse_sliding_direction,
        )

        LER = simulate_LER_batch(
            circuit, decoder, observable_matrix, num_trials, trial_seed
        )
        results.append(LER)
        print(f"{direction}, p: {p:.3f}, LER: {LER:.6f}")

print("Min-Sum (own impl):")
for p in ps:
    code, circuit = build_bb_72_circuit(p, num_rounds)

    model = circuit.detector_error_model(decompose_errors=False)
    check_matrix, observable_matrix, priors = detector_error_model_to_matrix(model)

    H_csr = csr_matrix(check_matrix)
    for direction, reverse_sliding_direction, trial_seed, results in (
        ("forward", False, seed + 1, LERs_own_min_sum_forward),
        ("backward", True, seed + 3, LERs_own_min_sum_backward),
    ):
        decoder = SlidingWindowMinSumDecoder(
            H=H_csr,
            m=code.hz.shape[0],
            num_rounds=num_rounds,
            W=W,
            F=F,
            priors=priors,
            max_iter=max_iter,
            warm_start=False,
            alpha=1.0,
            reverse_sliding_direction=reverse_sliding_direction,
        )

        LER = simulate_LER_batch(
            circuit, decoder, observable_matrix, num_trials, trial_seed
        )
        results.append(LER)
        print(f"{direction}, p: {p:.3f}, LER: {LER:.6f}")

print("SPA (Roffe et al. package):")
for p in ps:
    code, circuit = build_bb_72_circuit(p, num_rounds)

    decoder_params = {
        "bp_method": "product_sum",
        "max_iter": max_iter,
        "schedule": "parallel",
    }

    LER = apply_windowing_and_simulate_LER(
        code,
        circuit,
        LdpcDecoder,
        decoder_params,
        W,
        F,
        num_trials,
        seed,
    )

    LERs_roffe_spa.append(LER)
    print(f"p: {p:.3f}, LER: {LER:.6f}")

print("Min-Sum (Roffe et al. package):")
for p in ps:
    code, circuit = build_bb_72_circuit(p, num_rounds)

    decoder_params = {
        "bp_method": "min_sum",
        "max_iter": max_iter,
        "schedule": "parallel",
    }

    LER = apply_windowing_and_simulate_LER(
        code,
        circuit,
        LdpcDecoder,
        decoder_params,
        W,
        F,
        num_trials,
        seed,
    )

    LERs_roffe_min_sum.append(LER)
    print(f"p: {p:.3f}, LER: {LER:.6f}")

plt.figure()

for results, label in (
    (LERs_roffe_spa, "SPA [Roffe et al., forward]"),
    (LERs_own_spa_forward, "SPA [own, forward]"),
    (LERs_own_spa_backward, "SPA [own, backward]"),
    (LERs_roffe_min_sum, "Min-sum [Roffe et al., forward]"),
    (LERs_own_min_sum_forward, "Min-sum [own, forward]"),
    (LERs_own_min_sum_backward, "Min-sum [own, backward]"),
):
    LERs = np.asarray(results)
    LERs_per_round = 1 - (1 - LERs) ** (1 / num_rounds)

    # Binomial distribution / num_trials
    sigma = np.sqrt(LERs * (1 - LERs) / num_trials)
    sigma_per_round = 1 / num_rounds * (1 - LERs) ** (1 / num_rounds - 1) * sigma

    plt.errorbar(
        ps,
        LERs_per_round,
        yerr=3 * sigma_per_round,
        label=label,
        capsize=4,
    )

plt.yscale("log")
plt.title("Logical Error Rate for [72,12,6] BB Code Under Circuit Level Noise")
plt.xlabel("Physical Error Rate")
plt.ylabel("Logical Error Rate (LER)")
plt.grid()
plt.legend()
plt.show()
