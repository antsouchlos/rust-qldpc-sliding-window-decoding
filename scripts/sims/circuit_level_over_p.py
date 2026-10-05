from coordinator import Run, sweep

from rust_qldpc import (
    AscedMinSumDecoder,
    SlidingWindowAscedMinSumDecoder,
    SlidingWindowMinSumDecoder,
    StandardMinSumDecoder,
)
from sim import make_LER_evaluator

#
#
# Helper functions
#
#


def make_sw_asced(params):
    def make(H, priors, code, num_rounds):
        return SlidingWindowAscedMinSumDecoder(
            H=H,
            m=code.hz.shape[0],
            num_rounds=num_rounds,
            priors=priors,
            **params,
        )

    return make


def make_sw_min_sum(params):
    def make(H, priors, code, num_rounds):
        return SlidingWindowMinSumDecoder(
            H=H,
            m=code.hz.shape[0],
            num_rounds=num_rounds,
            priors=priors,
            **params,
        )

    return make


def make_min_sum(params):
    def make(H, priors, code, num_rounds):
        return StandardMinSumDecoder(
            H=H,
            priors=priors,
            **params,
        )

    return make

def make_asced(params):
    def make(H, priors, code, num_rounds):
        return AscedMinSumDecoder(
            H=H,
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

LER_settings = {"num_trials": num_trials, "seed": seed, "target_num_errors": 200}

ps = [0.001, 0.002, 0.003, 0.004, 0.005]

general_params = {"max_iter": 128, "alpha": 1.0}
window_params = {"W": 5, "F": 1}
asced_params = {"delta": 2, "num_batches": 4, "splitter_weight": 30}

#
#
# Simulation
#
#

runs = [
    Run(
        f"sw-asced-ms_cold_W{window_params['W']}-F{window_params['F']}-d{asced_params['delta']}-b{asced_params['num_batches']}-sw{asced_params['splitter_weight']}",
        make_LER_evaluator(make_sw_asced, **LER_settings),
        {
            **general_params,
            **window_params,
            **asced_params,
            "warm_start": False,
        },
    ),
    Run(
        f"asced-ms_warm_W{window_params['W']}-F{window_params['F']}-d{asced_params['delta']}-b{asced_params['num_batches']}-sw{asced_params['splitter_weight']}",
        make_sw_asced,
        {
            **general_params,
            **window_params,
            **asced_params,
            "warm_start": True,
        },
    ),
    Run(
        f"sw-ms_cold_W{window_params['W']}-F{window_params['F']}",
        make_sw_min_sum,
        {**general_params, **window_params, "warm_start": False},
    ),
    Run(
        f"sw-ms_warm_W{window_params['W']}-F{window_params['F']}",
        make_sw_min_sum,
        {**general_params, **window_params, "warm_start": True},
    ),
    Run(
        "ms",
        make_min_sum,
        {**general_params},
    ),
    Run(
        "asced-ms",
        make_LER_evaluator(make_asced, **LER_settings),
        {**general_params, **asced_params},
    ),
]

sweep(
    runs,
    axis="p",
    axis_values=ps,
    experiment=experiment,
    shared={"num_rounds": num_rounds},
    meta=LER_settings,
)
