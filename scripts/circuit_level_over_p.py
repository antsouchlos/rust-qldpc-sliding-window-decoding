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
num_trials = 10

ps = [0.001, 0.002, 0.003, 0.004, 0.005]

window = {"W": 5, "F": 1, "max_iter": 100}
splitters = {"delta": 2, "num_batches": 6, "splitter_weight": 4}

#
#
# Simulation
#
#

# fmt: off
runs = [
    Run("asced-ms_cold_W5-F1-d2-b6", make_asced, {**window, **splitters, "warm_start": False, "alpha": 1.0}),
    Run("asced-ms_warm_W5-F1-d2-b6", make_asced, {**window, **splitters, "warm_start": True, "alpha": 1.0}),
    Run("sw-ms_W5-F1", make_min_sum, {**window, "warm_start": False, "alpha": 1.0}),
]
# fmt: on

sweep(
    runs,
    axis="p",
    axis_values=ps,
    experiment=experiment,
    circuit_params={"num_rounds": num_rounds},
    num_trials=num_trials,
    seed=seed,
)
