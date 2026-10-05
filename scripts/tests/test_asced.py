import matplotlib.pyplot as plt
import numpy as np
from quits import BbCode
from scipy.sparse import csr_matrix
from tqdm import tqdm

from rust_qldpc import AscedSpaDecoder

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


def simulate_LER_batch(
    H: csr_matrix,
    L: csr_matrix,
    p: float,
    decoder: AscedSpaDecoder,
    max_trials: int,
    seed: int,
    target_errors: int,
    batch_size: int = 64,
):
    rng = np.random.default_rng(seed)

    H_dense = H.todense()
    _, n = H_dense.shape

    logical_errors = 0
    total_trials = 0

    pbar = tqdm(total=target_errors)
    while total_trials < max_trials:
        current_batch_size = min(batch_size, max_trials - total_trials)
        if current_batch_size <= 0:
            break

        # Compute error patterns and syndromes

        errors = (rng.random((current_batch_size, n)) < p).astype(np.uint8)
        syndromes = (H @ errors.T % 2).T.astype(np.uint8)

        # Decode

        estimates = decoder.decode_batch(syndromes)

        # Check for logical errors

        residuals = errors ^ estimates
        num_errors = (np.any((L @ residuals.T) % 2, axis=0)).sum()

        total_trials += current_batch_size
        logical_errors += num_errors
        pbar.update(num_errors)

        if logical_errors >= target_errors:
            break

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

    priors = np.zeros(code.hx.shape[1]) + p
    channel_llrs = np.log((1 - priors) / priors)

    H_csr = csr_matrix(code.hx)
    L_csr = csr_matrix(code.lx)

    decoder = AscedSpaDecoder(
        H_csr,
        priors=priors,
        delta=delta,
        splitter_weight=splitter_weight,
        max_iter=max_iter,
        num_batches=num_splitter_batches,
    )

    LER, num_trials = simulate_LER_batch(
        H_csr, L_csr, p, decoder, max_trials, seed, target_errors
    )
    LERs.append(LER)
    trials.append(num_trials)
    print(f"p: {p:.3f}, LER: {LER:.6f}")

LERs = np.array(LERs)
trials = np.array(trials)
sigma = np.sqrt(LERs * (1 - LERs) / trials)

print(sigma)


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
    yerr=3 * sigma,
    label="Own results",
    capsize=4,
)

plt.yscale("log")
plt.title(
    "Logical error rate for [[144,12,12]] BB code under depolarizing "
    "noise and aSCED decoding"
)
plt.xlabel("Physical Error Rate")
plt.ylabel("Logical Error Rate (LER)")
plt.grid()
plt.legend()
plt.show()
