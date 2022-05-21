RPC_HOST="172.16.2.241"
RPC_PORT="4445"
RPC_KEY="rtx-3080"
RPC_WORKERS=1
TARGET="nvidia/geforce-rtx-3080"
LOG_DIR=$HOME/logs/ms-cpu-$1/
NUM_TRIALS=2000
CMD="tvm.meta_schedule.testing.tune_te_meta_schedule"

mkdir -p $LOG_DIR

run () {
    name=$1
    WORK_DIR=$LOG_DIR/$name/
    mkdir -p $WORK_DIR
    echo "Running workload $name @ $LOG_DIR"
    python -m $CMD                          \
        --workload "$name"                  \
        --target "$TARGET"                  \
        --work-dir "$WORK_DIR"              \
        --rpc-host "$RPC_HOST"              \
        --rpc-port "$RPC_PORT"              \
        --rpc-key "$RPC_KEY"                \
        --rpc-workers "$RPC_WORKERS"        \
        --num-trials $NUM_TRIALS            \
        2>&1 | tee "$WORK_DIR/$name.log"
}

# run C1D
# run C2D
# run C3D
# run CAP
# run DEP
# run DIL
# run GMM
# run GRP
# run T2D
# run C2d-BN-RELU
# run TBG
# run NRM
# run SFM
# run WINOGRAD
run WINOGRAD_WO_RULES