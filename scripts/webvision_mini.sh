export PYTHONWARNINGS="ignore"

# SEED
export SEED=0
# GPU
export GPU=$1
# Dataset
export DATASET='mini_webvision'
export CONFIG="config/${DATASET}.yaml"
export DELTA=0.00
# Hyper-parameters
export ALPHA=0.2
export BETA=0.1
export GAMMA=0.005
export N=10
export TOPK=5
export TOPKD=1000
export TAUM=0.99
# Log Group Tag
export LOGPROJ="benchmark_miniWebVision"

export LOGNAME="resnet50-topK${TOPK}_decay${TOPKD}-Knn${N}-ALPHA${ALPHA}-BETA${BETA}-GAMMA${GAMMA}-DELTA${DELTA}-TAUM${TAUM}-NoConfWeight"
CUDA_VISIBLE_DEVICES=${GPU} python main.py --seed ${SEED} \
        --cfg ${CONFIG} --gpu 0 --log-proj ${LOGPROJ} --log-name ${LOGNAME} \
        --arch resnet50 --opt sgd --batch-size 64 --lr 0.01 --warmup-lr 0.01 --lr-decay step:30,60,80 --weight-decay 1e-4 --eps 0.5 --epochs 100 \
        --warmup-epochs 5 --fdim 512 --hdim 2 --queue-length 32000 --knet-m 0.99 --transform strong \
        --alpha ${ALPHA} --beta ${BETA} --gamma ${GAMMA} --n-neighbors ${N} --topK ${TOPK} --topK-decay ${TOPKD} --delta ${DELTA} --tau-m ${TAUM} \
        --cls4ood nl --cls4id ce --ncr-lossfun kldiv --integrate-mode or --ood-criterion div --threshold-generator per_class_mean \
        --save-model --enable-progress-bar True --eval-det 0 --benchmark --warmup-fc-only   # --conf-weight

export LOGNAME="inception_resnet_v2-topK${TOPK}_decay${TOPKD}-Knn${N}-ALPHA${ALPHA}-BETA${BETA}-GAMMA${GAMMA}-DELTA${DELTA}-TAUM${TAUM}-NoConfWeight"
CUDA_VISIBLE_DEVICES=${GPU} python main.py --seed ${SEED} \
        --cfg ${CONFIG} --gpu 0 --log-proj ${LOGPROJ} --log-name ${LOGNAME} \
        --arch InceptionResNetV2 --opt sgd --batch-size 64 --lr 0.01 --warmup-lr 0.01 --lr-decay step:20,40,60,80 --weight-decay 1e-4 --eps 0.5 --epochs 100 \
        --warmup-epochs 5 --fdim 512 --hdim 2 --queue-length 32000 --knet-m 0.99 --transform strong \
        --alpha ${ALPHA} --beta ${BETA} --gamma ${GAMMA} --n-neighbors ${N} --topK ${TOPK} --topK-decay ${TOPKD} --delta ${DELTA} --tau-m ${TAUM} \
        --cls4ood nl --cls4id ce --ncr-lossfun kldiv --integrate-mode or --ood-criterion div --threshold-generator per_class_mean \
        --save-model --enable-progress-bar True --eval-det 0 --benchmark --warmup-fc-only   # --conf-weight
