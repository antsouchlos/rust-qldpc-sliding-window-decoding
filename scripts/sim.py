import functools
from collections.abc import Callable

import numpy as np
import stim
from quits import CircuitBuildOptions, ErrorModel
from quits.decoder import detector_error_model_to_matrix
from quits.qldpc_code import BbCode
from quits.simulation import get_stim_mem_result
from scipy.sparse import csc_matrix, csr_matrix
from tqdm import tqdm

CIRCUIT_PARAMS = ("p", "num_rounds")


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


@functools.lru_cache(maxsize=1)
def build_circuit_and_matrices(p: float, num_rounds: int):
    code, stim_circuit = build_bb_72_circuit(p, num_rounds)

    model = stim_circuit.detector_error_model(decompose_errors=False)
    check_matrix, observable_matrix, priors = detector_error_model_to_matrix(model)

    H = csr_matrix(check_matrix)
    priors = np.asarray(priors, dtype=np.float64)

    return code, stim_circuit, H, observable_matrix, priors


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


def derive_LER_statistics(LER: float, num_trials: int, num_rounds: int) -> dict:
    sigma = np.sqrt(LER * (1 - LER) / num_trials)
    survival = 1 - LER
    per_round_derivative = survival ** (1 / num_rounds - 1) if survival > 0 else 0.0

    return {
        "LER_3sigma": float(3 * sigma),
        "LER_per_round": float(1 - survival ** (1 / num_rounds)),
        "LER_per_round_3sigma": float(3 * per_round_derivative * sigma / num_rounds),
    }


def make_LER_evaluator(
    make_decoder: Callable,
    num_trials: int,
    seed: int,
    target_num_errors: int,
    batch_size_target: int = 1000,
) -> Callable[[dict], dict]:
    def evaluate(point: dict) -> dict:
        num_rounds = point["num_rounds"]
        code, stim_circuit, H, observable_matrix, priors = build_circuit_and_matrices(
            point["p"], num_rounds
        )

        decoder_params = {
            key: value for key, value in point.items() if key not in CIRCUIT_PARAMS
        }
        decoder = make_decoder(decoder_params)(H, priors, code, num_rounds)

        LER, total_trials = simulate_LER_batch(
            stim_circuit,
            decoder,
            observable_matrix,
            num_trials,
            seed,
            batch_size_target=batch_size_target,
            target_num_errors=target_num_errors,
        )

        return {
            "LER": float(LER),
            "num_trials": total_trials,
            **derive_LER_statistics(LER, total_trials, num_rounds),
        }

    return evaluate
