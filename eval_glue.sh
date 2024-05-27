TASKS=$1 
echo $TASKS
export PYTHONPATH=/fairseq-0.9.0
ROBERTA_PATH=/mnt/vol1/gaowenchun/new/roberta_test/roberta-model/checkpoints/checkpoint_test.pt
#TASKS="QQP MNLI QNLI MRPC RTE STS-B SST-2 CoLA"

if [ "$TASKS" = "QQP" ]
then
    TOTAL_NUM_UPDATES=113272 # 10 epochs through RTE for bsz 16
    WARMUP_UPDATES=28318     # 6 percent of the number of updates
    LR=1e-05                # Peak LR for polynomial LR scheduler.
    NUM_CLASSES=2
    MAX_SENTENCES=32       # Batch size.
    TASK_DIR=QQP-bin/ 
elif [ "$TASKS" = "MNLI" ]
then
    TOTAL_NUM_UPDATES=123873  # 10 epochs through RTE for bsz 16
    WARMUP_UPDATES=7432      # 6 percent of the number of updates
    LR=1e-5            # Peak LR for polynomial LR scheduler.
    NUM_CLASSES=3
    MAX_SENTENCES=32        # Batch size.`
    TASK_DIR=MNLI-bin/ 
elif [ "$TASKS" = "QNLI" ]
then
    echo $TASKS
    echo "1111111111111"
    TOTAL_NUM_UPDATES=33112  # 10 epochs through RTE for bsz 16
    WARMUP_UPDATES=1986      # 6 percent of the number of updates
    LR=1e-5            # Peak LR for polynomial LR scheduler.
    NUM_CLASSES=2
    MAX_SENTENCES=32        # Batch size.
    TASK_DIR=QNLI-bin/
elif [ "$TASKS" = "MRPC" ]
then
    TOTAL_NUM_UPDATES=2296  # 10 epochs through RTE for bsz 16
    WARMUP_UPDATES=137      # 6 percent of the number of updates
    LR=1e-5            # Peak LR for polynomial LR scheduler.
    NUM_CLASSES=2
    MAX_SENTENCES=16        # Batch size.
    TASK_DIR=MRPC-bin/ 
elif [ "$TASKS" = "RTE" ]
then
    TOTAL_NUM_UPDATES=2036  # 10 epochs through RTE for bsz 16
    WARMUP_UPDATES=122      # 6 percent of the number of updates
    LR=2e-5            # Peak LR for polynomial LR scheduler.
    NUM_CLASSES=2
    MAX_SENTENCES=16        # Batch size.
    TASK_DIR=RTE-bin/
elif [ "$TASKS" = "SST-2" ]
then
    TOTAL_NUM_UPDATES=20935  # 10 epochs through RTE for bsz 16
    WARMUP_UPDATES=1256      # 6 percent of the number of updates
    LR=1e-5            # Peak LR for polynomial LR scheduler.
    NUM_CLASSES=2
    MAX_SENTENCES=32        # Batch size.
    TASK_DIR=SST-2-bin/
elif [ "$TASKS" = "COLA" ]
then
    TOTAL_NUM_UPDATES=5336  # 10 epochs through RTE for bsz 16
    WARMUP_UPDATES=320      # 6 percent of the number of updates
    LR=1e-5            # Peak LR for polynomial LR scheduler.
    NUM_CLASSES=2
    MAX_SENTENCES=16        # Batch size.
    TASK_DIR=CoLA-bin/
elif [ "$TASKS" = "STS-B" ]
then
    TOTAL_NUM_UPDATES=3598  # 10 epochs through RTE for bsz 16
    WARMUP_UPDATES=214      # 6 percent of the number of updates
    LR=2e-5            # Peak LR for polynomial LR scheduler.
    NUM_CLASSES=6
    MAX_SENTENCES=16        # Batch size.
    TASK_DIR=STS-B-bin/
fi

CUDA_VISIBLE_DEVICES=0 python train.py $TASK_DIR \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_12layer_768hidden_12head_2048ffn \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    #--save-dir checkpoints \
    # --regression-target