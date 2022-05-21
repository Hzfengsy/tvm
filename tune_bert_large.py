import pickle
import torch
import tvm
from tvm import meta_schedule as ms
import tvm.relay.testing
import tvm.contrib.graph_executor as runtime

from bert_rewrite import rewrite_reshape_gelu

import logging
import numpy as np
from typing import Tuple, List

import tvm
from tvm import relay
from tvm.ir import IRModule
from tvm.runtime.ndarray import cpu, cuda
from tvm.target.target import Target
from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor
from tvm.meta_schedule.database import PyDatabase, Workload, TuningRecord, JSONDatabase
from tvm.meta_schedule.tune import (
    tune_relay,
)
from tvm.meta_schedule import ApplyHistoryBest, extract_task_from_relay
from tvm.meta_schedule import schedule_rule as M
from tvm.meta_schedule import postproc
from tvm.meta_schedule.utils import derived_object
from tvm.script import tir as T
from tvm import tir


logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)

with open("models/bert_large.json", "r") as fi:
    mod = tvm.ir.load_json(fi.read())
with open("models/bert_large.params", "rb") as fi:
    params = relay.load_param_dict(fi.read())

mod = rewrite_reshape_gelu(mod)
target = tvm.target.Target("nvidia/geforce-rtx-3070")
num_trials = 20000
work_dir = "~/logs/ms-bert-large"


def build_relay(database):
    with ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            return tvm.relay.build(mod, target=target, params=params)


def tune():
    # extract workloads from relay program
    print("Extract tasks...")

    # run tuning tasks
    print("Tuning...")
    rpc_config = ms.runner.RPCConfig(
        tracker_host="192.168.6.66",
        tracker_port=4445,
        tracker_key="3070",
        session_timeout_sec=3600,
    )
    runner = ms.runner.RPCRunner(
        rpc_config=rpc_config,
        evaluator_config=ms.runner.EvaluatorConfig(
            number=3,
            repeat=1,
            min_repeat_ms=100,
            enable_cpu_cache_flush=False,
        ),
    )
    return ms.tune_relay(
        mod=mod,
        target=target,
        config=ms.TuneConfig(
            strategy="evolutionary",
            num_trials_per_iter=64,
            max_trials_per_task=num_trials,
            max_trials_global=num_trials,
        ),
        runner=runner,  # type: ignore
        work_dir=work_dir,
        params=params,
    )


def evaluate(lib):
    dev = tvm.device(str(target), 0)

    module = runtime.GraphModule(lib["default"](dev))

    batch_size = 8
    seq_len = 128
    torch.manual_seed(1001)
    inputs = (
        torch.randint(high=100, size=(batch_size, seq_len), dtype=torch.int64),
        torch.randint(high=100, size=(batch_size, seq_len), dtype=torch.int64),
        torch.randint(high=100, size=(batch_size, seq_len), dtype=torch.int64),
    )
    pickle.dump(inputs, open("inputs.pkl", "wb"))
    inputs = pickle.load(open("inputs.pkl", "rb"))

    module.set_input("input_ids", inputs[0])
    module.set_input("attention_mask", inputs[1])
    module.set_input("token_type_ids", inputs[2])
    module.run()
    print(module.get_output(0))
    out = module.get_output(0).numpy()
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, number=1, repeat=50))


lib = tune()
evaluate(lib)
