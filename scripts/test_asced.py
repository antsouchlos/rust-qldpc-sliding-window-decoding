import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from quits import BbCode
from scipy.sparse import csr_matrix
from tqdm import tqdm

from rust_qldpc import AscedMinSumDecoder

# %%


def build_bb_code(N: int):
    # fmt: off
    if N == 72:
        code = BbCode(l=6, m=6, A_x_pows=[3], A_y_pows=[1, 2], B_x_pows=[1, 2], B_y_pows=[3])
    elif N == 90:
        code = BbCode(l=15, m=3, A_x_pows=[9], A_y_pows=[1, 2], B_x_pows=[2, 7], B_y_pows=[0])
    elif N == 108:
        code = BbCode(l=9, m=6, A_x_pows=[3], A_y_pows=[1, 2], B_x_pows=[1, 2], B_y_pows=[3])
    elif N == 144:
        code = BbCode(l=12, m=6, A_x_pows=[3], A_y_pows=[1, 2], B_x_pows=[1, 2], B_y_pows=[3])
    elif N == 288:
        code = BbCode(l=12, m=12, A_x_pows=[3], A_y_pows=[2, 7], B_x_pows=[1, 2], B_y_pows=[3])
    elif N == 360:
        code = BbCode(l=30, m=6, A_x_pows=[9], A_y_pows=[1, 2], B_x_pows=[25, 26], B_y_pows=[3])
    elif N == 756:
        code = BbCode(l=21, m=18, A_x_pows=[3], A_y_pows=[10, 17], B_x_pows=[3, 19], B_y_pows=[5])
    else:
        assert False, "Unsupported code size"
    # fmt: on

    return code


@nb.jit(nopython=True)
def gf2_rank(A: np.ndarray) -> int:
    M = A.copy()
    m, n = M.shape
    rank = 0
    col = 0

    while col < n and rank < m:
        pivot = np.argmax(M[rank:, col]) + rank

        if M[pivot, col] == 0:
            col += 1
            continue

        if pivot != rank:
            tmp = M[rank].copy()
            M[rank] = M[pivot]
            M[pivot] = tmp

        for r in range(m):
            if r != rank and M[r, col]:
                M[r] ^= M[rank]

        rank += 1
        col += 1

    return rank


@nb.jit(nopython=True, parallel=True)
def check_if_in_row_space(errors, estimates, H_dense, rank_H):
    """
    True for each (error, estimate) whose residual lies in the PCM rowspace.

    When the residual `r = error ^ estimate` lies in the PCM rowspace, it is a
    stabilizer, i.e., the estimate only differs from the actual error by a
    stabilizer. Decoding was thus successful.
    """
    batch = errors.shape[0]
    m, n = H_dense.shape
    in_rowspace = np.empty(batch, dtype=np.bool_)

    for i in nb.prange(batch):
        residual = errors[i] ^ estimates[i]
        A = np.empty((m + 1, n), dtype=np.uint8)
        A[:m] = H_dense
        A[m] = residual
        in_rowspace[i] = gf2_rank(A) == rank_H

    return in_rowspace


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
    p: float,
    decoder: AscedMinSumDecoder,
    max_trials: int,
    seed: int,
    target_errors: int,
    batch_size: int = 64,
):
    np.random.seed(seed)

    p_err = p
    # p_err = 2.0 * p / 3.0

    H_dense = H.todense()
    _, n = H_dense.shape
    rank_H = gf2_rank(H_dense)

    # errors = (np.random.random((max_trials, n)) < p).astype(np.uint8)
    # s = compute_syndrome_batch(errors, H.indices, H.indptr, m)
    #
    # s_split = s.reshape((100, -1, s.shape[1]))

    logical_errors = 0
    total_trials = 0

    while total_trials < max_trials:
        current_batch_size = min(batch_size, max_trials - total_trials)
        if current_batch_size <= 0:
            break

        # Compute error pattern

        errors = np.random.binomial(1, p_err, (current_batch_size, n)).astype(np.uint8)
        syndromes = (H @ errors.T).T.astype(np.uint8) % 2

        # Decode

        estimates = decoder.decode_batch(syndromes)

        # Check for logical errors

        in_rowspace = check_if_in_row_space(errors, estimates, H_dense, rank_H)

        total_trials += current_batch_size
        logical_errors += (~in_rowspace).sum()

        if logical_errors >= target_errors:
            break

    # for s_batch in tqdm(s_split):
    #     if num_errors >= target_errors:
    #         break
    #
    #     e_hat = decoder.decode_batch(s_batch)
    #     s_hat = compute_syndrome_batch(e_hat, H.indices, H.indptr, m)
    #
    #     in_rowspace = check_if_in_row_space(errors, e_hat, H_dense, rank_H)
    #
    #     total_trials += current_batch_size
    #     logical_errors += (~in_rowspace).sum()
    #
    #     # TODO: This is not the logical error rate
    #     num_errors += np.sum(((s_batch - s_hat) % 2).any(axis=1))
    #     num_trials += s_batch.shape[0]

    LER = logical_errors / total_trials

    return LER, total_trials


# %%


N = 144
seed = 2
max_trials = 100000
target_errors = 200
delta = 2
splitter_weight = 4
max_iter = 100
num_splitter_batches = 4

ps = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

print(f"Ensemble size: {num_splitter_batches * (1 << delta)}")

LERs = []
trials = []

for p in ps:
    code = build_bb_code(N)

    priors = np.zeros(code.hz.shape[1]) + p
    channel_llrs = np.log((1 - priors) / priors)

    H_csr = csr_matrix(code.hz)

    decoder = AscedMinSumDecoder(
        H_csr,
        priors=priors,
        delta=delta,
        splitter_weight=splitter_weight,
        max_iter=max_iter,
        num_batches=num_splitter_batches,
    )

    LER, num_trials = simulate_LER_batch(
        H_csr, p, decoder, max_trials, seed, target_errors
    )
    LERs.append(LER)
    trials.append(num_trials)
    print(f"p: {p:.3f}, LER: {LER:.6f}")

LERs = np.array(LERs)
jrials = np.array(trials)
sigma = np.sqrt(LERs * (1 - LERs) / trials)


LERs_original_impl = [
    0.008648308302005593,
    0.031005304369790352,
    0.08447729672650475,
    0.1904382470119522,
    0.30900409276944063,
    0.5,
    0.6267880364109233,
]

plt.plot(ps, LERs_original_impl, label="Original implementation")
plt.errorbar(
    ps,
    LERs,
    yerr=sigma,
    label="Own results",
    capsize=4,
)

plt.yscale("log")
plt.title("Logical error rate for ")
plt.xlabel("Physical Error Rate")
plt.ylabel("Logical Error Rate (LER)")
plt.grid()
plt.legend()
plt.show()
