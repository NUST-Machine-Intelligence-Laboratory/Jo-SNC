export PYTHONWARNINGS="ignore"

# SEED
export SEED=0
# GPU
export GPU=$1

# Dataset
export DATASET='aircraft'
export CONFIG="config/${DATASET}.yaml"
export DELTA=0.0
# Hyper-parameters
export ALPHA=0.3
export BETA=0.8
export GAMMA=0.001
export N=10
export TOPK=5
export TOPKD=1000
export TAU_M=0.98
# Log Group Tag
export LOGPROJ="benchmark_webfg"
export LOGNAME="resnet50-topK${TOPK}_decay${TOPKD}-Knn${N}-ALPHA${ALPHA}-BETA${BETA}-GAMMA${GAMMA}-DELTA${DELTA}-TAUM${TAU_M}"

CUDA_VISIBLE_DEVICES=${GPU} python main.py --seed ${SEED} \
        --cfg ${CONFIG} --gpu 0 --log-proj ${LOGPROJ} --log-name ${LOGNAME} \
        --arch resnet50 --opt sgd --batch-size 32 --lr 0.005 --warmup-lr 0.005 --lr-decay cosine:10,1e-8,110 --weight-decay 3e-4 --eps 0.3 --epochs 120 \
        --warmup-epochs 5 --fdim 512 --hdim 2 --queue-length 32000 --knet-m 0.99 --transform strong \
        --alpha ${ALPHA} --beta ${BETA} --gamma ${GAMMA} --n-neighbors ${N} --topK ${TOPK} --topK-decay ${TOPKD} --delta ${DELTA} --tau-m ${TAU_M} \
        --cls4ood nl --cls4id ce --ncr-lossfun kldiv --integrate-mode or --ood-criterion div --threshold-generator per_class_mean \
        --save-model --enable-progress-bar True --eval-det 0 --benchmark --conf-weight
