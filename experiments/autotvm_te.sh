RPC_HOST="172.16.2.241"
RPC_PORT="4445"
RPC_KEY="rtx-3080"
RPC_WORKERS=1
TARGET="nvidia/geforce-rtx-3080"
LOG_DIR=$HOME/logs/autotvm-cuda-$1/
CMD="tvm.meta_schedule.testing.tune_te_autotvm"

mkdir -p $LOG_DIR

run () {
  workload="$1"
  input_shape="$2"
  num_trials="$3"
  WORK_DIR=$LOG_DIR/$workload/
  mkdir -p $WORK_DIR

  python -m $CMD                      \
    --wkl "$workload"                 \
    --batch-size 1                    \
    2>&1 | tee -a "$WORK_DIR/$workload.log"
}

run "bert_base"     "[1,64]"          2000
run "resnet_50"     "[1,3,224,224]"   2000
run "mobilenet_v2"  "[1,3,224,224]"   2000