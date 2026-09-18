from rust_qldpc import (
    SlidingWindowAscedMinSumDecoder,
    SlidingWindowMinSumDecoder,
    StandardMinSumDecoder,
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


def make_min_sum(params):
    def make(H, priors, code, num_rounds):
        return StandardMinSumDecoder(
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


experiment = "circuit-level-asced-exploration"

seed = 1
num_rounds = 6
num_trials = 1000000

p_fixed = 0.003

window_params = {"W": 5, "F": 1, "max_iter": 100, "warm_start": False}
asced_params = {"delta": 2, "num_batches": 4, "splitter_weight": 4}

independent_vars = {
    "splitter_weight": [50, 40, 30, 20, 10, 6, 4, 2],
    # "num_batches": [1, 2, 4, 6, 8],
    # "delta": [1, 2, 4, 6],
}


#
#
# Simulation
#
#


# fmt: off
runs = [
]
# fmt: on

for axis, axis_values in independent_vars.items():
    sweep(
        [
            Run(
                f"asced-ms_cold_{axis}",
                make_sw_asced,
                {**window_params, **asced_params, "alpha": 1.0},
            )
        ],
        axis=axis,
        axis_values=axis_values,
        experiment=experiment,
        circuit_params={"p": p_fixed, "num_rounds": num_rounds},
        num_trials=num_trials,
        seed=seed,
    )
