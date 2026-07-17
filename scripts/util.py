import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

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
                raise ValueError("syndromes must be a 2-D array (num_trials, num_checks)")

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
            if name.startswith('_'):
                raise AttributeError(name)
            return getattr(self._local_decoder, name)

    return _Wrapper
