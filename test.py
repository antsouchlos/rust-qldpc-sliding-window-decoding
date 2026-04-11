# %%


import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import stim
from tqdm import tqdm
import matplotlib.pyplot as plt

from quits import ErrorModel, CircuitBuildOptions
from quits.qldpc_code import BbCode
from quits.simulation import get_stim_mem_result
from quits.decoder import detector_error_model_to_matrix

from ldpc.bp_decoder import BpDecoder as LdpcDecoder
from rust_qldpc import SyndromeMinSumDecoder, SyndromeSpaDecoder
# from py3s.bp import BpDecoder as Py3sDecoder


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


# %%


seed = 1
num_rounds = 2
num_trials = 10000
max_iter = 30

# ps = [0.001, 0.002, 0.003, 0.004, 0.005]
ps = [0.001]


LERs_py3s_spa = []
LERs_py3s_min_sum = []
LERs_ldpc_spa = []
LERs_ldpc_min_sum = []

print("SPA (own impl):")
for p in ps:
    code, circuit = build_bb_72_circuit(p, num_rounds)

    model = circuit.detector_error_model(decompose_errors=False)
    check_matrix, observable_matrix, priors = detector_error_model_to_matrix(model)

    priors = np.array(priors)
    channel_llrs = np.log((1 - priors) / priors)

    H_csr = csr_matrix(check_matrix)
    decoder = SyndromeSpaDecoder(
        H_csr.indptr,
        H_csr.indices,
        H_csr.data,
        H_csr.shape,
        channel_llrs=channel_llrs,
        max_iter=max_iter,
    )

    LER = simulate_LER(code, circuit, decoder, observable_matrix, num_trials, seed)
    LERs_py3s_spa.append(LER)
    print(f"p: {p:.3f}, LER: {LER:.6f}")

# print("Min-Sum (own impl):")
# for p in ps:
#     code, circuit = build_bb_72_circuit(p, num_rounds)
#
#     model = circuit.detector_error_model(decompose_errors=False)
#     check_matrix, observable_matrix, priors = detector_error_model_to_matrix(model)
#
#     priors = np.array(priors)
#     channel_llrs = np.log((1 - priors) / priors)
#
#     H_csr = csr_matrix(check_matrix)
#     decoder = SyndromeMinSumDecoder(
#         H_csr.indptr,
#         H_csr.indices,
#         H_csr.data,
#         H_csr.shape,
#         channel_llrs=channel_llrs,
#         max_iter=max_iter,
#     )
#
#     LER = simulate_LER(code, circuit, decoder, observable_matrix, num_trials, seed + 1)
#     LERs_py3s_min_sum.append(LER)
#     print(f"p: {p:.3f}, LER: {LER:.6f}")

print("SPA (ldpc package):")
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
    LERs_ldpc_spa.append(LER)
    print(f"p: {p:.3f}, LER: {LER:.6f}")
#
# print("Min-Sum (ldpc package):")
# for p in ps:
#     code, circuit = build_bb_72_circuit(p, num_rounds)
#
#     model = circuit.detector_error_model(decompose_errors=False)
#     check_matrix, observable_matrix, priors = detector_error_model_to_matrix(model)
#
#     decoder = LdpcDecoder(
#         check_matrix,
#         bp_method="min_sum",
#         max_iter=max_iter,
#         schedule="parallel",
#         channel_probs=priors,
#     )
#
#     LER = simulate_LER(code, circuit, decoder, observable_matrix, num_trials, seed + 3)
#     LERs_ldpc_min_sum.append(LER)
#     print(f"p: {p:.3f}, LER: {LER:.6f}")
#
# LERs_py3s_spa = np.array(LERs_py3s_spa)
# LERs_py3s_min_sum = np.array(LERs_py3s_min_sum)
# LERs_ldpc_spa = np.array(LERs_ldpc_spa)
# LERs_ldpc_min_sum = np.array(LERs_ldpc_min_sum)
#
# plt.figure()
#
# for LERs, label in zip(
#     [LERs_py3s_spa, LERs_py3s_min_sum, LERs_ldpc_spa, LERs_ldpc_min_sum],
#     [
#         "SPA [own]",
#         "Min-sum [own]",
#         "SPA [ldpc package]",
#         "Min-sum [ldpc package]",
#     ],
# ):
#     LERs_per_round = 1 - (1 - LERs) ** (1 / num_rounds)
#
#     # Binomial distribution / num_trials
#     sigma = np.sqrt(LERs * (1 - LERs) / num_trials)
#     sigma_per_round = 1 / num_rounds * (1 - LERs) ** (1 / num_rounds - 1) * sigma
#
#     plt.errorbar(
#         ps,
#         LERs_per_round,
#         yerr=sigma_per_round,
#         label=label,
#         capsize=4,
#     )
#
# plt.yscale("log")
# plt.title("Logical Error Rate for [72,12,6] BB Code Under Circuit Level Noise")
# plt.xlabel("Physical Error Rate")
# plt.ylabel("Logical Error Rate (LER)")
# plt.grid()
# plt.legend()
# plt.show()
#
#
# # %% [markdown]
#
# """
# ## Outdated code: Find syndromes producing different results and investigate decoder behavior
# """
#
# # %%
#
#
# # def get_logical_predictions(code, observable_matrix, detection_events, decoder):
# #     num_trials = detection_events.shape[0]
# #     logical_pred = np.zeros((num_trials, code.lz.shape[0]), dtype=int)
# #
# #     for i in tqdm(range(num_trials)):
# #         syndrome = (detection_events[i, :].copy()) % 2
# #
# #         e_hat = decoder.decode(syndrome)
# #         logical_pred[i, :] = observable_matrix @ e_hat % 2
# #
# #     return logical_pred
# #
# #
# # # %%
# #
# #
# # # Set params
# #
# # seed = 1
# # num_rounds = 1
# # num_trials = 1000
# # max_iter = 30
# # p = 0.001
# #
# # # Prepare circuit and samples
# #
# # code, circuit = build_bb_72_circuit(p, num_rounds)
# # model = circuit.detector_error_model(decompose_errors=False)
# # check_matrix, observable_matrix, priors = detector_error_model_to_matrix(model)
# #
# # detection_events, observable_flips = get_stim_mem_result(circuit, num_trials, seed=seed)
# #
# # # Create logical predictions from decoders
# #
# # bp_decoder = BpDecoder(
# #     check_matrix,
# #     bp_method="product_sum",
# #     max_iter=max_iter,
# #     schedule="parallel",
# #     channel_probs=priors,
# # )
# # bp_predictions = get_logical_predictions(
# #     code, observable_matrix, detection_events, bp_decoder
# # )
# #
# # spa_decoder = SPADecoder(
# #     check_matrix, max_iter=max_iter, channel_probs=np.array(priors)
# # )
# # spa_predictions = get_logical_predictions(
# #     code, observable_matrix, detection_events, spa_decoder
# # )
# #
# # # Find diffent predictions
# #
# # trial_logicals_different = np.any(bp_predictions != spa_predictions, axis=1)
# #
# # diff_idx = np.argwhere(trial_logicals_different)[0]
# #
# # diff_detection_events = detection_events[diff_idx, :]
# # diff_observable_flips = observable_flips[diff_idx, :]
# #
# # print(f"diff_detection_events = {diff_detection_events * 1}")
# # print(f"diff_observable_flips = {diff_observable_flips * 1}")
# #
# #
# # # %% [markdown]
# #
# # """
# # ## Investigate difference in behavior
# # """
# #
# # # %%
# #
# #
# # np.set_printoptions(linewidth=np.inf, precision=5)
# #
# # syndrome = diff_detection_events[0, :]
# #
# # print(f"check_matrix.shape: {check_matrix.shape}", flush=True)
# # print(f"syndrome:         {syndrome * 1}", flush=True)
# # print(f"observable_flips: {diff_observable_flips * 1}", flush=True)
# #
# # print(
# #     "================================ LDPC package ================================",
# #     flush=True,
# # )
# #
# # e_hat_bp = bp_decoder.decode(syndrome)
# # logical_pred = observable_matrix @ e_hat_bp % 2
# # err = ((observable_flips - logical_pred) % 2).any()
# #
# # print(f"e_hat:            {e_hat_bp * 1}", flush=True)
# # print(f"logical_pred:     {logical_pred * 1}", flush=True)
# # print(f"err:              {err * 1}", flush=True)
# #
# # print(
# #     "================================== Own impl ==================================",
# #     flush=True,
# # )
# #
# # e_hat_spa = spa_decoder.decode(syndrome)
# # logical_pred = observable_matrix @ e_hat_spa % 2
# # err = ((observable_flips - logical_pred) % 2).any()
# #
# # print(f"e_hat:            {e_hat_spa * 1}", flush=True)
# # print(f"logical_pred:     {logical_pred * 1}", flush=True)
# # print(f"err:              {err * 1}", flush=True)
# #
# # print(
# #     "==============================================================================",
# #     flush=True,
# # )
# #
# # print(f"Found same e_hat: {np.all(e_hat_bp == e_hat_spa)}", flush=True)
