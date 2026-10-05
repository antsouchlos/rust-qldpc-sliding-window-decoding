import functools
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import stim
from quits import CircuitBuildOptions, ErrorModel
from quits.decoder import detector_error_model_to_matrix
from quits.qldpc_code import BbCode
from quits.simulation import get_stim_mem_result
from scipy.sparse import csc_matrix, csr_matrix
from tqdm import tqdm

#
#
# Alist parsing
#
#


def _parse_alist_header(header):
    size = header.split()
    return int(size[0]), int(size[1])


def read_alist_file(filename):
    """
    This function reads in an alist file and creates the corresponding parity
    check matrix H. The format of alist files is described at:
    http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html
    """

    with open(filename) as myfile:
        data = myfile.readlines()
        num_cols, num_rows = _parse_alist_header(data[0])

        H = np.zeros((num_rows, num_cols))

        # The locations of 1s starts in the 5th line of the file
        for line_number in np.arange(4, 4 + num_cols):
            indices = data[line_number].split()
            for index in indices:
                H[int(index) - 1, line_number - 4] = 1

        return H.astype(np.int32)


#
#
# Parallel batch decoder
#
#


def _make_parallel_decoder(cls, args, kwargs):
    return cls(*args, **kwargs)


_parallel_worker_decoder = None  # global because decoder instances may be unpicklable


def _init_parallel_worker(cls, args, kwargs):
    global _parallel_worker_decoder
    _parallel_worker_decoder = _make_parallel_decoder(cls, args, kwargs)


def _decode_parallel_chunk(chunk):
    global _parallel_worker_decoder
    if _parallel_worker_decoder is None:
        raise RuntimeError("Worker decoder not initialized")
    results = [_parallel_worker_decoder.decode(s) for s in chunk]
    return np.array(results)


def make_parallel_batch_decoder(decoder_cls):
    class _Wrapper:
        def __init__(self, *args, num_workers=None, **kwargs):
            self._args = args
            self._kwargs = kwargs
            cpu = os.cpu_count() or 1
            self._num_workers = cpu if num_workers is None else max(1, int(num_workers))
            self._local_decoder = decoder_cls(*args, **kwargs)
            self._pool = None

        def decode(self, syndrome):
            return self._local_decoder.decode(syndrome)

        def _get_pool(self):
            if self._pool is None:
                self._pool = ProcessPoolExecutor(
                    max_workers=self._num_workers,
                    initializer=_init_parallel_worker,
                    initargs=(decoder_cls, self._args, self._kwargs),
                )
            return self._pool

        def decode_batch(self, syndromes):
            syndromes = np.asarray(syndromes)
            if syndromes.ndim != 2:
                raise ValueError(
                    "syndromes must be a 2-D array (num_trials, num_checks)"
                )

            num_trials = syndromes.shape[0]
            if num_trials == 0:
                return np.empty((0, syndromes.shape[1]), dtype=np.uint8)

            if num_trials == 1 or self._num_workers <= 1:
                return np.array([self.decode(s) for s in syndromes])

            n_workers = min(self._num_workers, num_trials)
            chunks = np.array_split(syndromes, n_workers, axis=0)

            pool = self._get_pool()
            futures = [pool.submit(_decode_parallel_chunk, chunk) for chunk in chunks]
            results = [f.result() for f in futures]
            return np.concatenate(results, axis=0)

        def close(self):
            if self._pool is not None:
                self._pool.shutdown()
                self._pool = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return getattr(self._local_decoder, name)

    return _Wrapper


#
#
# Simulation helpers
#
#

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


def derive_ler_statistics(LER: float, num_trials: int, num_rounds: int) -> dict:
    sigma = np.sqrt(LER * (1 - LER) / num_trials)
    survival = 1 - LER
    per_round_derivative = survival ** (1 / num_rounds - 1) if survival > 0 else 0.0

    return {
        "LER_3sigma": float(3 * sigma),
        "LER_per_round": float(1 - survival ** (1 / num_rounds)),
        "LER_per_round_3sigma": float(3 * per_round_derivative * sigma / num_rounds),
    }


def make_ler_evaluator(
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
            **derive_ler_statistics(LER, total_trials, num_rounds),
        }

    return evaluate
