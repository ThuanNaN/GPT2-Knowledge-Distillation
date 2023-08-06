#!/usr/bin/bash
python train_adamw.py --wandb-log --wandb-project mle2 --wandb-run-name student_drop0 \
                        --save-dir shakespeare_student \
                        --dataset shakespeare \
                        --log-interval 10 \
                        --eval-interval 20 \
                        --eval-iters 500 \
                        --batch-size 2 \
                        --accumulation-steps 16 \
                        --block-size 1024 \
                        --num-layer 8 \
                        --num-head 8 \
                        --num-embd 1024 \
                        --dropout 0.1 \
                        --max-iters 1000 \
                        --learning-rate 1e-3 \
                        --decay-lr \
                        --lr-decay-iters 2000 \
                        --min-lr 1e-4 \
                        --beta2 0.95 \
                        --warmup-iters 200 
