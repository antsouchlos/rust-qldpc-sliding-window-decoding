import numba as nb
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

from rust_qldpc import (
    SyndromeMinSumDecoder,
    SyndromeSpaDecoder,
    SyndromeSpaGdDecoder,
)

from sw_gdg.codes_q import create_cyclic_permuting_matrix, create_QC_GHP_codes

# %%


@nb.njit(cache=True)
def compute_syndrome_batch(errors, H_col_indices, H_indptr, m):
    """
    Compute binary syndrome s = H * error mod 2 for a batch of error vectors.

    errors: 2D array of shape (batch_size, n), each row is an error vector.
    Returns: 2D array of shape (batch_size, m).
    """
    batch_size = errors.shape[0]
    syndromes = np.zeros((batch_size, m), dtype=np.uint8)
    for b in range(batch_size):
        for i in range(m):
            s = 0
            for idx in range(H_indptr[i], H_indptr[i + 1]):
                s ^= errors[b, H_col_indices[idx]]
            syndromes[b, i] = s
    return syndromes


def simulate_LER_batch(
    H: csr_matrix,
    decoder,
    num_trials: int,
    seed: int,
):
    np.random.seed(seed)

    m, n = H.shape  # type: ignore

    errors = (np.random.random((num_trials, n)) < p).astype(np.uint8)
    s = compute_syndrome_batch(errors, H.indices, H.indptr, m)

    e_hat = decoder.decode_batch(s)
    s_hat = compute_syndrome_batch(e_hat, H.indices, H.indptr, m)

    LER = np.mean(((s - s_hat) % 2).any(axis=1))

    return LER


# %% [markdown]

"""
## BP
"""

# %%


N = 144
seed = 2
num_trials_bp = 1000
num_trials_bpgd = 1000
max_iter_bp = 120
max_iter_bpgd = 100000
T = 1

ps = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

LERs_bp = []

for p in ps:
    code = create_QC_GHP_codes(
        63, create_cyclic_permuting_matrix(7, [27, 54, 0]), [0, 1, 6]
    )  # [[882,24,18<=d<=24]]

    priors = np.zeros(code.hz.shape[1]) + p
    channel_llrs = np.log((1 - priors) / priors)

    H_csr = csr_matrix(code.hz)
    decoder = SyndromeSpaDecoder(
        H_csr.indptr,
        H_csr.indices,
        H_csr.data,
        H_csr.shape,
        channel_llrs=channel_llrs,
        max_iter=max_iter_bp,
    )

    LER = simulate_LER_batch(H_csr, decoder, num_trials_bp, seed)
    LERs_bp.append(LER)
    print(f"p: {p:.3f}, LER: {LER:.6f}")

LERs_bpgd = []

for p in ps:
    code = create_QC_GHP_codes(
        63, create_cyclic_permuting_matrix(7, [27, 54, 0]), [0, 1, 6]
    )  # [[882,24,18<=d<=24]]

    priors = np.zeros(code.hz.shape[1]) + p
    channel_llrs = np.log((1 - priors) / priors)

    H_csr = csr_matrix(code.hz)
    decoder = SyndromeSpaGdDecoder(
        H_csr.indptr,
        H_csr.indices,
        H_csr.data,
        H_csr.shape,
        channel_llrs=channel_llrs,
        max_iter=max_iter_bpgd,
        T=T,
    )

    LER = simulate_LER_batch(H_csr, decoder, num_trials_bpgd, seed)
    LERs_bpgd.append(LER)
    print(f"p: {p:.3f}, LER: {LER:.6f}")


# %%


LERs_bp = np.array(LERs_bp)
sigma_bp = np.sqrt(LERs_bp * (1 - LERs_bp) / num_trials_bp)

LERs_bpgd = np.array(LERs_bpgd)
sigma_bpgd = np.sqrt(LERs_bpgd * (1 - LERs_bpgd) / num_trials_bpgd)

LERs_paper_bp = [
    0.04314399656081324,
    0.08535262790127975,
    0.1423767097403712,
    0.24935912794528423,
    0.4367285548529659,
    0.7109711253747334,
    0.9295099786844357,
]

LERs_paper_bpgd = [
    0.00002862798743422962,
    0.0003913744373720393,
    0.005994844115797479,
    0.05504795977792058,
    0.23462301630044863,
    0.5350509581355273,
    0.8431913451191565,
]

plt.plot(ps, LERs_paper_bp, label="BP: Yao et al.")
plt.errorbar(
    ps,
    LERs_bp,
    yerr=sigma_bp,
    label="BP: Own results",
    capsize=4,
)

plt.plot(ps, LERs_paper_bpgd, label="BPGD: Yao et al.")
plt.errorbar(
    ps,
    LERs_bpgd,
    yerr=sigma_bpgd,
    label="BPGD: Own results",
    capsize=4,
)

plt.yscale("log")
plt.title(
    "Logical Error Rate for [[882,24,18<=d<=24]]-GHP Code Under Bit-Flip Noise (SPA)"
)
plt.xlabel("Physical Error Rate")
plt.ylabel("Logical Error Rate (LER)")
plt.grid()
plt.legend()
plt.show()
