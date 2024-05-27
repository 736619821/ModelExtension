TASKS=$1 
echo $TASKS

if [ "$TASKS" = "ROBERTA" ]
then
    TOTAL_UPDATES=20000    # Total number of training steps
    PEAK_LR=0.0005        # Peak learning rate, adjust as needed
    TOKENS_PER_SAMPLE=512   # Max sequence length
    MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
    MAX_TOKENS=8192        # Number of sequences per batch (batch size)
    UPDATE_FREQ=16          # Increase the batch size 16x
    ARCH=roberta_6layer_384hidden_6head_1024ffn
    DATA_DIR=wiki_book_bin/

    python train.py --fp16 $DATA_DIR \
       --task masked_lm --criterion masked_lm \
		--save-dir roberta_checkpoints \
        --load-dir load_roberta_checkpoints \
	    --arch $ARCH \
		--dropout 0.1 \
		--optimizer adam --adam-betas '(0.9,0.98)' --weight-decay 0.01 --clip-norm 0.0 \
		--lr $PEAK_LR --lr-scheduler inverse_sqrt  --warmup-init-lr 1e-07 \
		--sample-break-mode none --tokens-per-sample $TOKENS_PER_SAMPLE \
		--max-tokens $MAX_TOKENS --update-freq $UPDATE_FREQ \
		--max-update $TOTAL_UPDATES --log-format json --log-interval 100 \
		--skip-invalid-size-inputs-valid-test \
		--fixed-validation-seed 0 \
		--ddp-backend no_c10d \
		--save-interval-updates 1000 \
		--reset-optimizer \
        --expand-layer

elif [ "$TASKS" = "GPT" ]
then
    TOTAL_UPDATES=20000   # Total number of training steps
    PEAK_LR=0.0005        # Peak learning rate, adjust as needed
    TOKENS_PER_SAMPLE=512   # Max sequence length
    MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
    MAX_TOKENS=8192        # Number of sequences per batch (batch size)
    UPDATE_FREQ=16          # Increase the batch size 16x
    ARCH=gpt_6layer_768hidden_6head_1024ffn
    DATA_DIR=wiki_book_bin/

    python train.py --fp16 $DATA_DIR \
        --task continual_lm --criterion continual_cross_entropy \
		--save-dir gpt_checkpoints \
		--load-dir load_gpt_checkpoints \
	    --arch $ARCH \
		--dropout 0.1 \
		--optimizer adam --adam-betas '(0.9,0.98)' --weight-decay 0.01 --clip-norm 0.0 \
		--lr $PEAK_LR --lr-scheduler inverse_sqrt  --warmup-init-lr 1e-07 \
		--sample-break-mode none --tokens-per-sample $TOKENS_PER_SAMPLE \
		--max-tokens $MAX_TOKENS --update-freq $UPDATE_FREQ \
		--max-update $TOTAL_UPDATES --log-format json --log-interval 100 \
		--skip-invalid-size-inputs-valid-test \
		--fixed-validation-seed 0 \
		--ddp-backend no_c10d \
		--save-interval-updates 1000 \
		--reset-optimizer \
        --expand-hidden

fi






