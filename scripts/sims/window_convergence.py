import numpy as np
from coordinator import Run, sweep
from tqdm import tqdm
from util import build_bb_code, build_circuit_and_matrices

from rust_qldpc import (
    AscedMinSumDecoder,
    StandardMinSumDecoder,
    get_window_borders,
)


def make_convergence_evaluator(
    windows,
    decoders,
    max_num_trials,
    batch_size,
    seed,
    W,
    reverse,
    target_num_errors,
    target_num_satisfied,
):
    def evaluate(point):
        window_index = point["window_index"] - 1
        H, priors = windows[window_index]
        decoder = decoders[window_index]
        window_seed = np.random.SeedSequence(
            [seed, W, int(reverse), point["window_index"]]
        )
        rng = np.random.default_rng(window_seed)
        satisfied = 0
        num_trials = 0

        description = (
            f"W={W} {point['direction']} window "
            f"{point['window_index']}/{len(windows)} {point['decoder']}"
        )
        with tqdm(
            total=max_num_trials, desc=description, unit="trial", unit_scale=True
        ) as progress:
            for start in range(0, max_num_trials, batch_size):
                count = min(batch_size, max_num_trials - start)

                errors = rng.binomial(1, priors, size=(count, len(priors))).astype(
                    np.uint8
                )
                syndromes = np.asarray((H @ errors.T % 2).T, dtype=np.uint8)

                estimates = decoder.decode_batch(syndromes)
                estimated_syndromes = np.asarray(
                    (H @ estimates.T % 2).T, dtype=np.uint8
                )

                satisfied += int(
                    np.count_nonzero(np.all(estimated_syndromes == syndromes, axis=1))
                )
                num_trials += count
                num_errors = num_trials - satisfied

                progress.update(count)
                progress.set_postfix(num_satisfied=satisfied, num_errors=num_errors)

                if (
                    num_errors >= target_num_errors
                    and satisfied >= target_num_satisfied
                ):
                    break

        return {
            "syndrome_satisfaction_rate": satisfied / num_trials,
            "num_satisfied": satisfied,
            "num_errors": num_trials - satisfied,
            "num_trials": num_trials,
        }

    return evaluate


def build_window_runs(
    H,
    priors,
    m,
    num_rounds,
    W,
    F,
    max_num_trials,
    min_sum_batch_size,
    asced_batch_size,
    seed,
    max_iter,
    asced_params,
    target_num_errors,
    target_num_satisfied,
):
    runs = []
    forward_borders = get_window_borders(H, m, num_rounds, W, F)
    reverse_borders = get_window_borders(H, m, num_rounds, W, F, True)

    for direction, reverse, borders in (
        ("forward", False, forward_borders),
        ("reverse", True, reversed(reverse_borders)),
    ):
        windows = []
        for (row_begin, col_begin), (row_end, col_end) in borders:
            window_matrix = H[row_begin : row_end + 1, col_begin : col_end + 1].tocsr()
            window_priors = priors[col_begin : col_end + 1]
            windows.append((window_matrix, window_priors))

        for decoder_name, decoder_class, decoder_settings in (
            ("min-sum", StandardMinSumDecoder, {}),
            ("asced-min-sum", AscedMinSumDecoder, asced_params),
        ):
            decoders = [
                decoder_class(
                    H=window_matrix,
                    priors=window_priors,
                    max_iter=max_iter,
                    alpha=1.0,
                    **decoder_settings,
                )
                for window_matrix, window_priors in windows
            ]
            runs.append(
                Run(
                    f"{direction}_W{W}_{decoder_name}",
                    make_convergence_evaluator(
                        windows,
                        decoders,
                        max_num_trials,
                        (
                            min_sum_batch_size
                            if decoder_class is StandardMinSumDecoder
                            else asced_batch_size
                        ),
                        seed,
                        W,
                        reverse,
                        target_num_errors,
                        target_num_satisfied,
                    ),
                    {
                        "direction": direction,
                        "W": W,
                        "F": F,
                        "decoder": decoder_name,
                        "alpha": 1.0,
                        **decoder_settings,
                    },
                )
            )

    return runs, len(forward_borders)


experiment = "isolated_window_convergence"

code_size = 144
p = 0.003
num_rounds = 6
max_num_trials = 1_000_000
asced_batch_size = 1000
min_sum_batch_size = 50 * asced_batch_size
seed = 1
max_iter = 128
target_num_errors = 200
target_num_satisfied = 200

window_sizes = [3, 4, 5, 6]
F = 1
asced_params = {"delta": 2, "num_batches": 4, "splitter_weight": 30}


code = build_bb_code(code_size)
_, _, H, _, priors = build_circuit_and_matrices(p, num_rounds, code)

for W in window_sizes:
    runs, num_windows = build_window_runs(
        H,
        priors,
        code.hz.shape[0],
        num_rounds,
        W,
        F,
        max_num_trials,
        min_sum_batch_size,
        asced_batch_size,
        seed,
        max_iter,
        asced_params,
        target_num_errors,
        target_num_satisfied,
    )

    sweep(
        runs,
        axis="window_index",
        axis_values=range(1, num_windows + 1),
        experiment=experiment,
        shared={
            "code_name": f"BB{code_size}",
            "p": p,
            "num_rounds": num_rounds,
            "max_iter": max_iter,
            "seed": seed,
        },
        meta={
            "max_num_trials": max_num_trials,
            "standard_min_sum_batch_size": min_sum_batch_size,
            "asced_min_sum_batch_size": asced_batch_size,
            "seed": seed,
            "target_num_errors": target_num_errors,
            "target_num_satisfied": target_num_satisfied,
        },
    )
