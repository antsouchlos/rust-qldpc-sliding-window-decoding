from rust_qldpc import (
    SlidingWindowAscedMinSumDecoder,
    SlidingWindowMinSumDecoder,
)
from sim import Run, sweep

#
#
# Helper functions
#
#


def make_asced(params):
    def make(H, priors, code, num_rounds):
        return SlidingWindowAscedMinSumDecoder(
            H=H,
            m=code.hz.shape[0],
            num_rounds=num_rounds,
            priors=priors,
            **params,
        )

    return make


def make_min_sum(params):
    def make(H, priors, code, num_rounds):
        return SlidingWindowMinSumDecoder(
            H=H,
            m=code.hz.shape[0],
            num_rounds=num_rounds,
            priors=priors,
            **params,
        )

    return make


#
#
# Params
#
#


experiment = "circuit-level-over-p"

seed = 1
num_rounds = 6
num_trials = 1000000

ps = [0.001, 0.002, 0.003, 0.004, 0.005]

window_params = {"W": 5, "F": 1, "max_iter": 100}
asced_params = {"delta": 2, "num_batches": 4, "splitter_weight": 4}

#
#
# Simulation
#
#

runs = [
    Run(
        f"asced-ms_cold_W{window_params['W']}-F{window_params['F']}-d{asced_params['delta']}-b{asced_params['num_batches']}-sw{asced_params['splitter_weight']}",
        make_asced,
        {**window_params, **asced_params, "warm_start": False, "alpha": 1.0},
    ),
    Run(
        f"asced-ms_warm_W{window_params['W']}-F{window_params['F']}-d{asced_params['delta']}-b{asced_params['num_batches']}-sw{asced_params['splitter_weight']}",
        make_asced,
        {**window_params, **asced_params, "warm_start": True, "alpha": 1.0},
    ),
    Run(
        f"sw-ms_cold_W{window_params['W']}-F{window_params['F']}",
        make_min_sum,
        {**window_params, "warm_start": False, "alpha": 1.0},
    ),
    Run(
        f"sw-ms_warm_W{window_params['W']}-F{window_params['F']}",
        make_min_sum,
        {**window_params, "warm_start": True, "alpha": 1.0},
    ),
]

sweep(
    runs,
    axis="p",
    axis_values=ps,
    experiment=experiment,
    circuit_params={"num_rounds": num_rounds},
    num_trials=num_trials,
    seed=seed,
)
