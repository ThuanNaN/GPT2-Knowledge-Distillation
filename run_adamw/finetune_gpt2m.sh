#!/usr/bin/bash
python train_adamw.py --wandb-log --wandb-project mle --wandb-run-name teacher \
                        --save-dir shakespeare-gpt2m \
                        --dataset shakespeare \
                        --log-interval 10 \
                        --eval-interval 20 \
                        --eval-iters 500 \
                        --batch-size 4 \
                        --accumulation-steps 16 \
                        --init-from gpt2-medium \
                        --dropout 0.1 \
                        --max-iters 500 \
                        --learning-rate 1e-4 