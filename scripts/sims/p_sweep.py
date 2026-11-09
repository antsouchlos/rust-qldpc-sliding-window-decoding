from coordinator import Run, sweep
from util import build_bb_code, make_ler_evaluator

from rust_qldpc import (
    AscedMinSumDecoder,
    SlidingWindowAscedMinSumDecoder,
    SlidingWindowMinSumDecoder,
    StandardMinSumDecoder,
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


def make_min_sum(params):
    def make(H, priors, code, num_rounds):
        return StandardMinSumDecoder(
            H=H,
            priors=priors,
            max_iter=params["max_iter"],
            alpha=params["alpha"],
        )

    return make


def make_asced(params):
    def make(H, priors, code, num_rounds):
        return AscedMinSumDecoder(
            H=H,
            priors=priors,
            max_iter=params["max_iter"],
            alpha=params["alpha"],
            delta=params["delta"],
            num_batches=params["num_batches"],
            splitter_weight=params["splitter_weight"],
        )

    return make


#
#
# Params
#
#


experiment = "forward_v_reverse_sliding"

seed = 1
num_rounds = 6
num_trials = 1_000_000
code_size = 144

eval_settings = {
    "num_trials": num_trials,
    "seed": seed,
    "target_num_errors": 1_000,
    "code": build_bb_code(code_size),
}

ps = [0.001, 0.002, 0.003, 0.004, 0.005]

general_params = {"max_iter": 128, "alpha": 1.0, "code_name": f"BB{code_size}"}
window_params = {"W": 3, "F": 1, "reverse_sliding_direction": False}
asced_params = {"delta": 2, "num_batches": 4, "splitter_weight": 30}

#
#
# Simulation
#
#

runs = [
    Run(
        "whole-min-sum",
        make_ler_evaluator(make_min_sum, **eval_settings),
        {**general_params},
    ),
    Run(
        (
            f"whole-asced-min-sum_d{asced_params['delta']}"
            f"-b{asced_params['num_batches']}"
            f"-sw{asced_params['splitter_weight']}"
        ),
        make_ler_evaluator(make_asced, **eval_settings),
        {**general_params, **asced_params},
    ),
    Run(
        f"sw-cold_forward_W{window_params['W']}-F{window_params['F']}",
        make_ler_evaluator(make_sw_min_sum, **eval_settings),
        {
            **general_params,
            **window_params,
            "warm_start": False,
            "reverse_sliding_direction": False,
        },
    ),
    Run(
        f"sw-warm_forward_W{window_params['W']}-F{window_params['F']}",
        make_ler_evaluator(make_sw_min_sum, **eval_settings),
        {
            **general_params,
            **window_params,
            "warm_start": True,
            "reverse_sliding_direction": False,
        },
    ),
    Run(
        f"sw-cold_backward_W{window_params['W']}-F{window_params['F']}",
        make_ler_evaluator(make_sw_min_sum, **eval_settings),
        {
            **general_params,
            **window_params,
            "warm_start": False,
            "reverse_sliding_direction": True,
        },
    ),
    Run(
        f"sw-warm_backward_W{window_params['W']}-F{window_params['F']}",
        make_ler_evaluator(make_sw_min_sum, **eval_settings),
        {
            **general_params,
            **window_params,
            "warm_start": True,
            "reverse_sliding_direction": True,
        },
    ),
    Run(
        f"sw-asced-cold_forward_W{window_params['W']}-F{window_params['F']}",
        make_ler_evaluator(make_sw_asced, **eval_settings),
        {
            **general_params,
            **window_params,
            **asced_params,
            "warm_start": False,
            "reverse_sliding_direction": False,
        },
    ),
    Run(
        f"sw-asced-warm_forward_W{window_params['W']}-F{window_params['F']}",
        make_ler_evaluator(make_sw_asced, **eval_settings),
        {
            **general_params,
            **window_params,
            **asced_params,
            "warm_start": True,
            "reverse_sliding_direction": False,
        },
    ),
    Run(
        f"sw-asced-cold_backward_W{window_params['W']}-F{window_params['F']}",
        make_ler_evaluator(make_sw_asced, **eval_settings),
        {
            **general_params,
            **window_params,
            **asced_params,
            "warm_start": False,
            "reverse_sliding_direction": True,
        },
    ),
    Run(
        f"sw-asced-warm_backward_W{window_params['W']}-F{window_params['F']}",
        make_ler_evaluator(make_sw_asced, **eval_settings),
        {
            **general_params,
            **window_params,
            **asced_params,
            "warm_start": True,
            "reverse_sliding_direction": True,
        },
    ),
]

sweep(
    runs,
    axis="p",
    axis_values=ps,
    experiment=experiment,
    shared={"num_rounds": num_rounds},
    meta={**eval_settings},
)
