# Optimize GPT2 model with knowledge distillation and quantization.

## 1. Optimize
### 1.1 Dataset
- [Tiny Shakespeare](https://huggingface.co/datasets/tiny_shakespeare)

### 1.2 Evaluation metric:
- [Perplexity](https://huggingface.co/docs/transformers/perplexity)

### 1.3 Result
All models is train or fintune with embedding length(emb) = 1024 and context length(ctx) = 1024.
#### Teacher:

| Model        | Layer | Head | Params | Size  |  loss  |
|:----------   |:----: |:----:| :----: | :---: |  :---: |
| GPT2-medium  |  24   |  16  |  354M  | 1.3GB |  3.036 |


#### Student:
| Model 	  | Layer | Head | Params 	| Size  |  loss   |             |
|:----------  | :----:|:----:|  :----:	| :---: |:-------:|:-----------:|
|       	  |       |      |          |       | Scratch | Distillation|
| GPT-student |   8   |  8   |  152M    |       |   4.95  |   4.7296    |


#### Checkpoint
- GPT2-medium: [Pre-trained](https://drive.google.com/file/d/1y7RYsqrGt7njagHAmGrlA2a6jseGwkGX/view?usp=drive_link)
- GPT-student: [Scratch](https://drive.google.com/file/d/191iLVLmueqbAodR0-prCZERNkpEu658p/view?usp=sharing) - [Distillation]()



## 2. Quantazation


### 3. Setup
#### 3.1 Install packages
```
pip install -r requirements.txt
```
#### 3.2 Download and prepare dataset
```
cd data/shakespeare
python prepare.py
```
#### 3.3 Training
```
bash run/finetune_gpt2m.sh
bash run/train_student.sh
bash run/train_student_distill.sh
```

## 4. Citation



