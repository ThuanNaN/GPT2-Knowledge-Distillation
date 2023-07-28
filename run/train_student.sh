
python train.py --save-dir shakespeare-char-baby \
                --dataset shakespeare_char \
                --log-interval 10 \
                --eval-interval 200 \
                --eval-iters 500 \
                --batch-size 64 \
                --accumulation_steps 1 \
                --block-size 256 \
                --num_layer 6 \
                --num_head 6 \
                --num_embd 384 \ 
                --dropout 0.2 \
                --max-iters 5000 \
                --learning-rate 1e-3 \
                --lr-decay-iters 5000 \
                --min_lr 1e-4 \
                --beta2 0.99 \
                --warmup-iters 200 
