import json
import logging
import math
import subprocess
import sys
from typing import Sequence

import numpy as np

import tvm
import tvm.rpc.tracker
import tvm.script
from tvm import relax
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.contrib.hexagon.tools import HEXAGON_SIMULATOR_NAME
from tvm.relax import register_pipeline
from tvm._ffi.runtime_ctypes import Device

logging.basicConfig(
    level=logging.INFO,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[{asctime}] {levelname}: {message}",
)

MODEL_PATH = "/home/syfeng/mlc-llm/debug/hexagon-final.py"
WEIGHT_PATH = "/home/syfeng/mlc-llm/dist/TinyLlama-1.1B-Chat-v1.0-q4f16_0-MLC"

target = tvm.target.hexagon("v73")
TARGET = tvm.target.Target(target, host=target)


@register_pipeline("empty")
def _empty_pipeline():
    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(
        mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext
    ) -> tvm.ir.IRModule:
        return mod

    return _pipeline


def get_adb_devices():
    result = subprocess.run(
        ["adb", "devices"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    output = result.stdout
    lines = output.strip().split("\n")[1:]

    devices = []
    for line in lines:
        if "unauthorized" in line or "offline" in line:
            continue
        if line.strip():
            device_info = line.split("\t")
            if len(device_info) == 2:
                devices.append(device_info[0])
    if len(devices) == 0:
        raise RuntimeError("No device found")
    return devices


def get_tracker_launcher():
    use_sim = True if "--simulator" in sys.argv[1:] else False

    tracker = tvm.rpc.tracker.Tracker("0.0.0.0", 9191)
    rpc_info = {
        "rpc_tracker_host": "172.16.2.219",
        "rpc_tracker_port": 9191,
        "adb_server_socket": "tcp:5037",
    }

    launcher = (
        HexagonLauncher(
            serial_number=HEXAGON_SIMULATOR_NAME,
            rpc_info=rpc_info,
            workspace="dist",
        )
        if use_sim
        else HexagonLauncher(
            serial_number=get_adb_devices()[0],
            rpc_info=rpc_info,
            sysmon_profile=False,
            clear_logcat=True,
        )
    )
    return tracker, launcher


def main():
    tracker, launcher = get_tracker_launcher()
    launcher.start_server()

    with launcher.create_session() as session:
        logging.info("Starting Building")
        mod = tvm.script.from_source(open(MODEL_PATH, encoding="utf-8").read())
        meta_data = str(mod["_metadata"].body.body.value)
        meta_data = json.loads(meta_data)
        exe = relax.build(mod, TARGET, pipeline="empty")
        logging.info("Build Finished")

        # logging.info("Starting Create Remote VM")
        dev = session.device
        # vm_mod = session.get_executor_from_factory(exe)
        # vm_rt = relax.VirtualMachine(vm_mod, dev)
        # logging.info("Create Remote VM Finished")

        logging.info("Start Loading Weight on Local")
        f_loader = tvm.get_global_func("vm.builtin.ndarray_cache.load")
        f_loader(WEIGHT_PATH, Device.kDLCPU, 0)
        f_loader = tvm.get_global_func("vm.builtin.param_array_from_cache_by_name")
        param_names: Sequence[str] = [params["name"] for params in meta_data["params"]]
        params = f_loader(param_names)
        logging.info("Loading Weight Finished on Local")

        logging.info("Start Pushing Weights to Remote")
        f_remote_update = session.get_function("vm.builtin.ndarray_cache.update")
        total_byte = 0
        for name, param in zip(param_names, params):
            param_bytes = math.prod(param.shape) * np.dtype(param.dtype).itemsize
            total_byte += param_bytes
            remote_array = tvm.nd.array(param, dev)
            f_remote_update(name, remote_array)
            logging.info(
                "Uploaded Params %s, size = %.2f MB, total size = %.2f MB",
                name,
                param_bytes / (2**20),
                total_byte / (2**20),
            )
        logging.info("Finish Pushing Weights to Remote")

        logging.info("Starting Run Initialize Effect")
        kv_cache = vm_rt["_initialize_effect"]()
        logging.info("Run Initialize Effect Finished")

        input_data = tvm.nd.array([[1]], device=dev)
        cache_len_shape = tvm.runtime.ShapeTuple([0])
        kv_seq_len_shape = tvm.runtime.ShapeTuple([0])
        cache_offset_shape = tvm.runtime.ShapeTuple([0])

        logging.info("Start Running Prefill")
        vm_rt["prefill"](
            input_data,
            cache_len_shape,
            kv_seq_len_shape,
            cache_offset_shape,
            kv_cache,
            kv_cache,
        )
        logging.info("Prefill Finished")

    launcher.stop_server()
    tracker.terminate()


if __name__ == "__main__":
    main()
