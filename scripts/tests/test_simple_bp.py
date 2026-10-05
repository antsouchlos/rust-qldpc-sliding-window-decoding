# %%


import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import stim
import matplotlib.pyplot as plt

from quits import ErrorModel, CircuitBuildOptions
from quits.qldpc_code import BbCode
from quits.simulation import get_stim_mem_result
from quits.decoder import detector_error_model_to_matrix

from ldpc.bp_decoder import BpDecoder as LdpcDecoder
from rust_qldpc import (
    StandardMinSumDecoder,
    StandardSpaDecoder,
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


def simulate_LER(
    code: BbCode,
    circuit: stim.Circuit,
    decoder,
    observable_matrix: csc_matrix,
    num_trials: int,
    seed: int,
):
    detection_events, observable_flips = get_stim_mem_result(
        circuit, num_trials, seed=seed
    )

    num_trials = detection_events.shape[0]
    logical_pred = np.zeros((num_trials, code.lz.shape[0]), dtype=int)

    for i in tqdm(range(num_trials)):
        syndrome = (detection_events[i, :].copy()) % 2

        e_hat = decoder.decode(syndrome)
        logical_pred[i, :] = observable_matrix @ e_hat % 2

    LER = np.mean(((observable_flips - logical_pred) % 2).any(axis=1))

    return LER


def simulate_LER_batch(
    code: BbCode,
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
num_rounds = 6
num_trials = 10000
max_iter = 32

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

    decoder = StandardSpaDecoder(
        csr_matrix(check_matrix),
        priors=np.array(priors, dtype=np.float64),
        max_iter=max_iter,
    )

    LER = simulate_LER_batch(
        code, circuit, decoder, observable_matrix, num_trials, seed
    )
    LERs_own_spa.append(LER)
    print(f"p: {p:.3f}, LER: {LER:.6f}")

print("Min-Sum (own impl):")
for p in ps:
    code, circuit = build_bb_72_circuit(p, num_rounds)

    model = circuit.detector_error_model(decompose_errors=False)
    check_matrix, observable_matrix, priors = detector_error_model_to_matrix(model)

    decoder = StandardMinSumDecoder(
        csr_matrix(check_matrix),
        priors=np.array(priors, dtype=np.float64),
        max_iter=max_iter,
        alpha=1.0,
    )

    LER = simulate_LER_batch(
        code, circuit, decoder, observable_matrix, num_trials, seed + 1
    )
    LERs_own_min_sum.append(LER)
    print(f"p: {p:.3f}, LER: {LER:.6f}")

print("SPA (Roffe et al. package):")
for p in ps:
    code, circuit = build_bb_72_circuit(p, num_rounds)

    model = circuit.detector_error_model(decompose_errors=False)
    check_matrix, observable_matrix, priors = detector_error_model_to_matrix(model)

    decoder = LdpcDecoder(
        check_matrix,
        bp_method="product_sum",
        max_iter=max_iter,
        schedule="parallel",
        channel_probs=priors,
    )

    LER = simulate_LER(code, circuit, decoder, observable_matrix, num_trials, seed + 2)
    LERs_roffe_spa.append(LER)
    print(f"p: {p:.3f}, LER: {LER:.6f}")

print("Min-Sum (Roffe et al. package):")
for p in ps:
    code, circuit = build_bb_72_circuit(p, num_rounds)

    model = circuit.detector_error_model(decompose_errors=False)
    check_matrix, observable_matrix, priors = detector_error_model_to_matrix(model)

    decoder = LdpcDecoder(
        check_matrix,
        bp_method="min_sum",
        max_iter=max_iter,
        schedule="parallel",
        channel_probs=priors,
    )

    LER = simulate_LER(code, circuit, decoder, observable_matrix, num_trials, seed + 3)
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
plt.title(
    "Logical Error Rate (Z Basis) for [72,12,6] BB Code Under Circuit Level Noise"
)
plt.xlabel("Physical Error Rate")
plt.ylabel("Logical Error Rate (LER)")
plt.grid()
plt.legend()
plt.show()
