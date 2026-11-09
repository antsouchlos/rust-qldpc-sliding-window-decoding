from coordinator import Run, sweep
from util import build_bb_code, make_ler_evaluator

from rust_qldpc import (
    SlidingWindowAscedMinSumDecoder,
    SlidingWindowMinSumDecoder,
)

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
            max_iter=params["max_iter"],
            alpha=params["alpha"],
            W=params["W"],
            F=params["F"],
            warm_start=params["warm_start"],
            reverse_sliding_direction=params["reverse_sliding_direction"],
            delta=params["delta"],
            num_batches=params["num_batches"],
            splitter_weight=params["splitter_weight"],
        )

    return make


def make_sw_min_sum(params):
    def make(H, priors, code, num_rounds):
        return SlidingWindowMinSumDecoder(
            H=H,
            m=code.hz.shape[0],
            num_rounds=num_rounds,
            priors=priors,
            max_iter=params["max_iter"],
            alpha=params["alpha"],
            W=params["W"],
            F=params["F"],
            warm_start=params["warm_start"],
            reverse_sliding_direction=params["reverse_sliding_direction"],
        )

    return make


#
#
# Params
#
#


experiment = "circuit_level_asced_param_exploration"
code = build_bb_code(72)

seed = 1
num_rounds = 6
num_trials = 1000000

LER_settings = {"num_trials": num_trials, "seed": seed, "target_num_errors": 200}

p_fixed = 0.003

general_params = {"max_iter": 128, "alpha": 1.0}
run_params = {**general_params, "code_name": f"BB{code.hx.shape[1]}"}
window_params = {
    "W": 5,
    "F": 1,
    "warm_start": False,
    "reverse_sliding_direction": False,
}
asced_params = {"delta": 2, "num_batches": 4, "splitter_weight": 8}

independent_vars = {
    # "splitter_weight": [600, 500, 400, 300, 200, 100, 50, 30, 20, 6, 4],
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
                make_ler_evaluator(make_sw_asced, code=code, **LER_settings),
                {**run_params, **window_params, **asced_params},
            )
        ],
        axis=axis,
        axis_values=axis_values,
        experiment=experiment,
        shared={"p": p_fixed, "num_rounds": num_rounds},
        meta=LER_settings,
    )
