import logging
from os import cpu_count

import tvm
from tvm.script import tir as T
import tir_tensor_intrin

from tvm import meta_schedule as ms
from tvm.meta_schedule import schedule_rule as M
from tvm.meta_schedule import postproc

num_trials = 2000
work_dir = "logs/bert_large_task"
target = tvm.target.Target("nvidia/geforce-rtx-3070")
rpc_config = ms.runner.RPCConfig(
    tracker_host="172.16.2.241",
    tracker_port=4445,
    tracker_key="rtx-3080",
    session_timeout_sec=60,
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

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.ERROR)


@T.prim_func
def task_func(
    placeholder: T.Buffer[(1024, 1024), "float16"],
    placeholder_1: T.Buffer[(4096, 1024), "float16"],
    placeholder_2: T.Buffer[(1, 4096), "float16"],
    placeholder_3: T.Buffer[(), "float16"],
    placeholder_4: T.Buffer[(), "float16"],
    placeholder_5: T.Buffer[(), "float16"],
    T_multiply: T.Buffer[(1024, 4096), "float16"],
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    # with T.block("root")
    T_dense = T.alloc_buffer([1024, 4096], dtype="float16")
    T_add = T.alloc_buffer([1024, 4096], dtype="float16")
    T_multiply_1 = T.alloc_buffer([1024, 4096], dtype="float16")
    T_cast = T.alloc_buffer([1024, 4096], dtype="float32")
    T_erf = T.alloc_buffer([1024, 4096], dtype="float32")
    T_cast_1 = T.alloc_buffer([1024, 4096], dtype="float16")
    T_multiply_2 = T.alloc_buffer([1024, 4096], dtype="float16")
    T_add_1 = T.alloc_buffer([1024, 4096], dtype="float16")
    for i0, i1, i2 in T.grid(1024, 4096, 1024):
        with T.block("T_dense"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            T.reads(placeholder[i, k], placeholder_1[j, k])
            T.writes(T_dense[i, j])
            T.block_attr(
                {
                    "workload": [
                        "dense_tensorcore.cuda",
                        ["TENSOR", [1024, 1024], "float16"],
                        ["TENSOR", [4096, 1024], "float16"],
                        None,
                        "float16",
                    ]
                }
            )
            with T.init():
                T_dense[i, j] = T.float16(0)
            T_dense[i, j] = T_dense[i, j] + placeholder[i, k] * placeholder_1[j, k]
    for i0, i1 in T.grid(1024, 4096):
        with T.block("T_add"):
            ax0, ax1 = T.axis.remap("SS", [i0, i1])
            T.reads(T_dense[ax0, ax1], placeholder_2[0, ax1])
            T.writes(T_add[ax0, ax1])
            T_add[ax0, ax1] = T_dense[ax0, ax1] + placeholder_2[0, ax1]
    for i0, i1 in T.grid(1024, 4096):
        with T.block("T_multiply"):
            ax0, ax1 = T.axis.remap("SS", [i0, i1])
            T.reads(T_add[ax0, ax1], placeholder_3[()])
            T.writes(T_multiply_1[ax0, ax1])
            T_multiply_1[ax0, ax1] = T_add[ax0, ax1] * placeholder_3[()]
    for i0, i1 in T.grid(1024, 4096):
        with T.block("T_cast"):
            ax0, ax1 = T.axis.remap("SS", [i0, i1])
            T.reads(T_multiply_1[ax0, ax1])
            T.writes(T_cast[ax0, ax1])
            T_cast[ax0, ax1] = T.cast(T_multiply_1[ax0, ax1], "float32")
    for i0, i1 in T.grid(1024, 4096):
        with T.block("T_erf"):
            ax0, ax1 = T.axis.remap("SS", [i0, i1])
            T.reads(T_cast[ax0, ax1])
            T.writes(T_erf[ax0, ax1])
            T_erf[ax0, ax1] = T.erf(T_cast[ax0, ax1], dtype="float32")
    for i0, i1 in T.grid(1024, 4096):
        with T.block("T_cast_1"):
            ax0, ax1 = T.axis.remap("SS", [i0, i1])
            T.reads(T_erf[ax0, ax1])
            T.writes(T_cast_1[ax0, ax1])
            T_cast_1[ax0, ax1] = T.cast(T_erf[ax0, ax1], "float16")
    for i0, i1 in T.grid(1024, 4096):
        with T.block("T_multiply_1"):
            ax0, ax1 = T.axis.remap("SS", [i0, i1])
            T.reads(T_cast_1[ax0, ax1], placeholder_4[()])
            T.writes(T_multiply_2[ax0, ax1])
            T_multiply_2[ax0, ax1] = T_cast_1[ax0, ax1] * placeholder_4[()]
    for i0, i1 in T.grid(1024, 4096):
        with T.block("T_add_1"):
            ax0, ax1 = T.axis.remap("SS", [i0, i1])
            T.reads(T_multiply_2[ax0, ax1], placeholder_5[()])
            T.writes(T_add_1[ax0, ax1])
            T_add_1[ax0, ax1] = T_multiply_2[ax0, ax1] + placeholder_5[()]
    for i0, i1 in T.grid(1024, 4096):
        with T.block("T_multiply_2"):
            ax0, ax1 = T.axis.remap("SS", [i0, i1])
            T.reads(T_add_1[ax0, ax1], T_add[ax0, ax1])
            T.writes(T_multiply[ax0, ax1])
            T_multiply[ax0, ax1] = T_add_1[ax0, ax1] * T_add[ax0, ax1]


def tune(level: int):
    # run tuning tasks
    print("Tuning...")

    def sch_rules():
        assert -1 < level < 5
        if level == 4:
            multi_level_tiling = M.MultiLevelTiling(
                structure="SSSRRSRS",
                tile_binds=["blockIdx.x", "blockIdx.y", "threadIdx.y"],
                use_tensor_core=True,
                max_innermost_factor=4,
                vector_load_lens=[1, 2, 4, 8],
                reuse_read=M.ReuseType(
                    req="must",
                    levels=[4],
                    scope="shared.dyn",
                ),
                reuse_write=M.ReuseType(
                    req="no",
                    levels=[3],
                    scope="shared.dyn",
                ),
            )
        else:
            multi_level_tiling = M.MultiLevelTiling(
                structure="SSSRRSRS",
                tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
                # use_tensor_core=(level >= 4),
                max_innermost_factor=64,
                vector_load_lens=[1, 2, 3, 4],
                reuse_read=M.ReuseType(
                    req="must",
                    levels=[4],
                    scope="shared",
                ),
                reuse_write=M.ReuseType(
                    req="must",
                    levels=[3],
                    scope="local",
                ),
            )
        auto_inline = M.AutoInline(
            into_producer=True,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=False,
            require_injective=False,
            require_ordered=False,
            disallow_op=None,
        )
        parallel = M.ParallelizeVectorizeUnroll(
            max_jobs_per_core=-1,  # disable parallelize
            max_vectorize_extent=-1,  # disable vectorize
            unroll_max_steps=[0, 16, 64, 512, 1024],
            unroll_explicit=True,
        )

        ret = []
        if level > 0:
            ret.append(multi_level_tiling)
        if level > 1:
            # Need twice
            # ret.append(auto_inline)
            ret.append(auto_inline)
        if level > 2:
            ret.append(parallel)
        return ret

    def postprocs():
        return [
            postproc.DisallowDynamicLoop(),
            postproc.RewriteCooperativeFetch(),
            postproc.RewriteUnboundBlock(),
            postproc.RewriteParallelVectorizeUnroll(),
            postproc.RewriteReductionBlock(),
            postproc.RewriteTensorCore(),
            postproc.VerifyGPUCode(),
        ]

    search_config = ms.TuneConfig(
        num_trials_per_iter=64,
        max_trials_per_task=num_trials,
        max_trials_global=num_trials,
        search_strategy_config={
            "population_size": 2048,
            "init_measured_ratio": 0.2,
            "init_min_unmeasured": 50,
            "genetic_num_iters": 3,
            "genetic_mutate_prob": 0.85,
            "genetic_max_fail_count": 10,
            "eps_greedy": 0.05,
        },
    )

    sch = ms.tune_tir(
        mod=task_func,
        target=target,
        config=search_config,
        sch_rules=sch_rules,
        postprocs=postprocs,
        work_dir=f"{work_dir}/{level}",
        num_threads=cpu_count(),
    )

    if sch is None:
        print("No valid schedule found!")
    else:
        print(sch.mod.script())
        print(sch.trace)


if __name__ == "__main__":
    for l in range(4, 5):
        print("level:", l)
        tune(l)
