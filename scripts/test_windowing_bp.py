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
    WindowingSyndromeMinSumDecoder,
    WindowingSyndromeSpaDecoder,
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

LERs_own_spa = []
LERs_own_min_sum = []
LERs_roffe_spa = []
LERs_roffe_min_sum = []

print("SPA (own impl):")
for p in ps:
    code, circuit = build_bb_72_circuit(p, num_rounds)

    model = circuit.detector_error_model(decompose_errors=False)
    check_matrix, observable_matrix, priors = detector_error_model_to_matrix(model)

    priors = np.array(priors)
    channel_llrs = np.log((1 - priors) / priors)

    H_csr = csr_matrix(check_matrix)
    decoder = WindowingSyndromeSpaDecoder(
        H_csr.indptr,
        H_csr.indices,
        H_csr.data,
        H_csr.shape,
        code.hz.shape[0],
        num_rounds,
        channel_llrs,
        W,
        F,
        pass_soft_info=False,
        max_iter=max_iter,
    )

    LER = simulate_LER_batch(circuit, decoder, observable_matrix, num_trials, seed)
    LERs_own_spa.append(LER)
    print(f"p: {p:.3f}, LER: {LER:.6f}")

print("Min-Sum (own impl):")
for p in ps:
    code, circuit = build_bb_72_circuit(p, num_rounds)

    model = circuit.detector_error_model(decompose_errors=False)
    check_matrix, observable_matrix, priors = detector_error_model_to_matrix(model)

    priors = np.array(priors)
    channel_llrs = np.log((1 - priors) / priors)

    H_csr = csr_matrix(check_matrix)
    decoder = WindowingSyndromeMinSumDecoder(
        H_csr.indptr,
        H_csr.indices,
        H_csr.data,
        H_csr.shape,
        code.hz.shape[0],
        num_rounds,
        channel_llrs,
        W,
        F,
        pass_soft_info=False,
        max_iter=max_iter,
    )

    LER = simulate_LER_batch(circuit, decoder, observable_matrix, num_trials, seed + 1)
    LERs_own_min_sum.append(LER)
    print(f"p: {p:.3f}, LER: {LER:.6f}")

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

LERs_own_spa = np.array(LERs_own_spa)
LERs_own_min_sum = np.array(LERs_own_min_sum)
LERs_roffe_spa = np.array(LERs_roffe_spa)
LERs_roffe_min_sum = np.array(LERs_roffe_min_sum)

plt.figure()

for LERs, label in zip(
    [LERs_roffe_spa, LERs_own_spa, LERs_roffe_min_sum, LERs_own_min_sum],
    [
        "SPA [Roffe et al.]",
        "SPA [own]",
        "Min-sum [Roffe et al.]",
        "Min-sum [own]",
    ],
):
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
