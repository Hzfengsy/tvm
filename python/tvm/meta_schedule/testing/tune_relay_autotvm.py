# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-docstring
import argparse
import json
import logging
import os


import numpy as np  # type: ignore
import tvm
from tvm import autotvm
from tvm import meta_schedule as ms
from tvm import relay
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.relay_workload import get_network


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--workload",
        type=str,
        required=True,
    )
    args.add_argument(
        "--input-shape",
        type=str,
        required=True,
    )
    args.add_argument(
        "--target",
        type=str,
        required=True,
    )
    args.add_argument(
        "--num-trials",
        type=int,
        required=True,
    )
    args.add_argument(
        "--rpc-host",
        type=str,
        required=True,
    )
    args.add_argument(
        "--rpc-port",
        type=int,
        required=True,
    )
    args.add_argument(
        "--rpc-key",
        type=str,
        required=True,
    )
    args.add_argument(
        "--rpc-workers",
        type=int,
        required=True,
    )
    args.add_argument(
        "--log-dir",
        type=str,
        required=True,
    )
    args.add_argument(
        "--cache-dir",
        type=str,
        default=True,
    )
    parsed = args.parse_args()
    parsed.target = tvm.target.Target(parsed.target)
    parsed.input_shape = json.loads(parsed.input_shape)
    parsed.rpc_config = ms.runner.RPCConfig(
        tracker_host=parsed.rpc_host,
        tracker_port=parsed.rpc_port,
        tracker_key=parsed.rpc_key,
        session_timeout_sec=3600,
    )
    return parsed


logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.INFO)
ARGS = _parse_args()


def main():
    log_file = os.path.join(ARGS.log_dir, f"{ARGS.workload}.json")
    mod, params, (input_name, input_shape, input_dtype) = get_network(
        ARGS.workload,
        ARGS.input_shape,
        cache_dir=ARGS.cache_dir,
    )
    print(f"Workload: {ARGS.workload}")
    print(f"  input_name: {input_name}")
    print(f"  input_shape: {input_shape}")
    print(f"  input_dtype: {input_dtype}")
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(mod["main"], target=ARGS.target, params=params)

    tuning_option = {
        "log_filename": log_file,
        "tuner": "xgb",
        "n_trial": ARGS.num_trials,
        "early_stopping": 600,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.RPCRunner(
                key=ARGS.rpc_key,
                host=ARGS.rpc_host,
                port=ARGS.rpc_port,
                number=3,
                repeat=1,
                min_repeat_ms=100,  # TODO
                enable_cpu_cache_flush=False,  # TODO
            ),
        ),
    }

    print("Tuning...")

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        tuner_obj = autotvm.tuner.XGBTuner(tsk, loss_type="rank")

        # do tuning
        tsk_trial = min(tuning_option["n_trial"], len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=tuning_option["early_stopping"],
            measure_option=tuning_option["measure_option"],
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_file),
            ],
        )

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=ARGS.target, params=params)

    graph, rt_mod, params = lib.graph_json, lib.lib, lib.params
    if input_dtype.startswith("float"):
        input_data = np.random.uniform(size=input_shape).astype(input_dtype)
    else:
        input_data = np.random.randint(low=0, high=10000, size=input_shape, dtype=input_dtype)

    def f_timer(rt_mod, dev, input_data):
        # pylint: disable=import-outside-toplevel
        from tvm.contrib.graph_executor import GraphModule

        # pylint: enable=import-outside-toplevel

        mod = GraphModule(rt_mod["default"](dev))
        mod.set_input(input_name, input_data)
        ftimer = mod.module.time_evaluator(
            "run",
            dev,
            min_repeat_ms=500,
            repeat=3,
        )
        results = list(np.array(ftimer().results) * 1000.0)  # type: ignore
        print("Running time in time_evaluator: ", results)

    run_module_via_rpc(
        rpc_config=ARGS.rpc_config,
        lib=lib,
        dev_type=ARGS.target.kind.name,
        args=[input_data],
        continuation=f_timer,
    )


if __name__ == "__main__":
    main()
