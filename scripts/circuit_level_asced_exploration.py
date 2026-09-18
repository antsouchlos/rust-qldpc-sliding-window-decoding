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


#
#
# Params
#
#


experiment = "circuit-level-asced-exploration"

seed = 1
num_rounds = 6
num_trials = 1000000

p_fixed = 0.003

general_params = {"max_iter": 128, "alpha": 1.0}
window_params = {"W": 5, "F": 1, "warm_start": False}
asced_params = {"delta": 2, "num_batches": 4, "splitter_weight": 30}

independent_vars = {
    "splitter_weight": [600, 500, 400, 300, 200, 100, 50, 30, 20, 6, 4],
    "num_batches": [1, 2, 4, 6, 8, 16, 32, 64],
    "delta": [1, 2, 4, 6],
    "max_iter": [32, 64, 100, 128, 256, 384, 512],
}


#
#
# Simulation
#
#


for axis, axis_values in independent_vars.items():
    sweep(
        [
            Run(
                f"asced-ms_cold_{axis}",
                make_sw_asced,
                {**general_params, **window_params, **asced_params},
            )
        ],
        axis=axis,
        axis_values=axis_values,
        experiment=experiment,
        circuit_params={"p": p_fixed, "num_rounds": num_rounds},
        num_trials=num_trials,
        seed=seed,
    )
