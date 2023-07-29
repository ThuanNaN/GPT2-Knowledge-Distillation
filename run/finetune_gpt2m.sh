#!/usr/bin/bash
python train_adamw.py --save-dir shakespeare-char-gpt2m \
                        --dataset shakespeare_char \
                        --log-interval 10 \
                        --eval-interval 50 \
                        --eval-iters 500 \
                        --batch-size 4 \
                        --accumulation-steps 16 \
                        --init-from gpt2-medium \
                        --dropout 0.1 \
                        --max-iters 500 \
                        --learning-rate 1e-4 