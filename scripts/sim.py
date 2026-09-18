from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import stim
from quits import CircuitBuildOptions, ErrorModel
from quits.decoder import detector_error_model_to_matrix
from quits.qldpc_code import BbCode
from quits.simulation import get_stim_mem_result
from scipy.sparse import csc_matrix, csr_matrix
from tqdm import tqdm

from results import open_run


# TODO: Replace this with arbitrary size BB code generation
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


def simulate_LER_batch(
    circuit: stim.Circuit,
    decoder,
    observable_matrix: csc_matrix,
    max_num_trials: int,
    seed: int,
    batch_size_target: int = 1000,
    target_num_errors: int = 200,
):
    logical_errors = 0
    total_trials = 0

    pbar = tqdm(total=max_num_trials)
    while total_trials < max_num_trials:
        batch_size = min(batch_size_target, max_num_trials - total_trials)

        # Compute syndromes for this batch via circuit-level noise

        detection_events, observable_flips = get_stim_mem_result(
            circuit, batch_size, seed=seed + total_trials
        )
        syndromes = np.asarray(detection_events, dtype=np.uint8)

        # Decode

        e_hats = decoder.decode_batch(syndromes)

        # Check for logical errors

        logical_pred = np.asarray((observable_matrix @ e_hats.T % 2).T, dtype=np.uint8)
        batch_errors = (observable_flips.astype(np.uint8) ^ logical_pred).any(axis=1)

        total_trials += batch_size
        logical_errors += batch_errors.sum()
        pbar.update(batch_size)

        # For error rates above 0.5, the number of successful trials is what
        # should reach the threshold
        if (logical_errors >= target_num_errors) and (
            total_trials - logical_errors >= target_num_errors
        ):
            break

    pbar.close()

    LER = logical_errors / total_trials
    return LER, total_trials


#
#
# Sweep driver
#
#


@dataclass
class Run:
    """
    One curve of a sweep.

    `label` names the run directory, `make_decoder` receives
    `(H, priors, code, num_rounds)` and `params` is recorded in full so that
    changing a parameter forks a new run directory.
    """

    label: str
    make_decoder: Callable
    params: dict = field(default_factory=dict)


def sweep(
    runs,
    ps,
    experiment: str,
    num_rounds: int,
    num_trials: int,
    seed: int,
    build_circuit: Callable = build_bb_72_circuit,
    batch_size_target: int = 1000,
    target_num_errors: int = 200,
):
    """Simulate every (run, p) that is not already present in its run directory."""

    meta = {
        "ps": list(ps),
        "num_rounds": num_rounds,
        "num_trials": num_trials,
        "seed": seed,
        "target_num_errors": target_num_errors,
    }
    run_dirs = {
        run.label: open_run(experiment, run.label, run.params, meta) for run in runs
    }

    for p in ps:
        pending = [run for run in runs if not run_dirs[run.label].has_p(p)]
        if not pending:
            print(f"p: {p:.4f}, all runs done")
            continue

        code, circuit = build_circuit(p, num_rounds)
        model = circuit.detector_error_model(decompose_errors=False)
        check_matrix, observable_matrix, priors = detector_error_model_to_matrix(model)

        H = csr_matrix(check_matrix)
        priors = np.asarray(priors, dtype=np.float64)

        for run in pending:
            decoder = run.make_decoder(H, priors, code, num_rounds)

            LER, total_trials = simulate_LER_batch(
                circuit,
                decoder,
                observable_matrix,
                num_trials,
                seed,
                batch_size_target=batch_size_target,
                target_num_errors=target_num_errors,
            )

            run_dirs[run.label].append(
                {
                    "p": p,
                    "LER": LER,
                    "num_trials": total_trials,
                    "num_rounds": num_rounds,
                }
            )
            print(
                f"{run.label}, p: {p:.4f}, LER: {LER:.6f}, num_trials: {total_trials}"
            )

    return {label: run_dir.path for label, run_dir in run_dirs.items()}
