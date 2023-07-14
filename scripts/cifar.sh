export PYTHONWARNINGS="ignore"

# SEED
export SEED=0
# GPU
export GPU=$1
# Dataset
export DATASET='cifar80n'
export NOISE_TYPE='sym'
export NOISE_RATE=50
export DELTA=0.0
export CONFIG="config/${DATASET}/${DATASET}-${NOISE_TYPE}${NOISE_RATE}.yaml"
# Hyper-parameters
export ALPHA=0.3
export BETA=0.1
export GAMMA=0.0001
export N=10
export TOPK=5
export TOPKD=1000
export TAU_M=0.975
# Log Group Tag
export LOGPROJ="ablation_study"

export LOGNAME="ModuleImpact_IDLearning_PllStyle"
CUDA_VISIBLE_DEVICES=${GPU} python cifar.py --seed ${SEED} \
        --cfg ${CONFIG} --gpu 0 --log-proj ${LOGPROJ} --log-name ${LOGNAME} \
        --arch cnn --opt adam --lr 0.001 --warmup-lr 0.001 --lr-decay cosine:10 --warmup-lr-plan constant --weight-decay 0 \
        --warmup-epochs 10 --fdim 256 --hdim 2 --queue-length 32000 --knet-m 0.99 --transform strong \
        --alpha ${ALPHA} --beta ${BETA} --gamma ${GAMMA} --n-neighbors ${N} --topK ${TOPK} --topK-decay ${TOPKD} --delta ${DELTA} --tau-m ${TAU_M} \
        --cls4ood nl --cls4id ce --ncr-lossfun kldiv --integrate-mode or --ood-criterion div --threshold-generator per_class_mean \
        --enable-progress-bar True --benchmark --conf-weight
