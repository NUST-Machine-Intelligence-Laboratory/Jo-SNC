export PYTHONWARNINGS="ignore"

# SEED
export SEED=0
# GPU
export GPU=$1
# Dataset
export DATASET='animal10n'
export CONFIG="config/${DATASET}.yaml"
export DELTA=0.0
# Hyper-parameters
export ALPHA=0.2
export BETA=0.1
export GAMMA=0.005
export N=10
export TOPK=5
export TOPKD=1000
export TAUM=0.99
# Log Group Tag
export LOGPROJ="benchmark_animal10n"

export LOGNAME="resnet50-topK${TOPK}_decay${TOPKD}-Knn${N}-ALPHA${ALPHA}-BETA${BETA}-GAMMA${GAMMA}-DELTA${DELTA}-TAUM${TAUM}-NoConfWeight"
CUDA_VISIBLE_DEVICES=${GPU} python main.py --seed ${SEED} \
        --cfg ${CONFIG} --gpu 0 --log-proj ${LOGPROJ} --log-name ${LOGNAME} \
        --arch vgg19_bn --opt sgd --batch-size 128 --lr 0.01 --warmup-lr 0.01 --lr-decay cosine:10 --weight-decay 5e-4 --eps 0.8 --epochs 100 \
        --warmup-epochs 5 --fdim 512 --hdim 2 --queue-length 32000 --knet-m 0.99 --transform strong \
        --alpha ${ALPHA} --beta ${BETA} --gamma ${GAMMA} --n-neighbors ${N} --topK ${TOPK} --topK-decay ${TOPKD} --delta ${DELTA} --tau-m ${TAUM} \
        --cls4ood nl --cls4id ce --ncr-lossfun kldiv --integrate-mode or --ood-criterion div --threshold-generator per_class_mean \
        --save-model --enable-progress-bar True --eval-det 0 --benchmark --warmup-fc-only   # --conf-weight
