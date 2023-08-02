# Optimize GPT2 model with knowledge distillation and quantization.

## 1. Optimize
### 1.1 Input 
#### GPT2-medium
```
num_layer: 24
num_head: 16
emb: 1024
```


#### GPT2-baby
```
num_layer: 6
num_head: 6
emb: 384
```


### 1.2 Evaluation metric:
- [Perplexity](https://huggingface.co/docs/transformers/perplexity)

### 1.3 Dataset
- [Tiny Shakespeare](https://huggingface.co/datasets/tiny_shakespeare)

### 1.4 Result
#### Teacher:
| Model        | Params | Size  |  loss  |
|:----------   | :----: | :---: |  :---: |
| GPT2-medium  |  354M  | 1.3GB |  3.036 |


#### Student:
| Model 	  | Params 	| Size   | loss    |             |
|:----------  | :----:	| :---:  |:-------:|:-----------:|
|       	  |        	|        | Scratch | Distillation|
| GPT-student |      |    |   |             |


#### Checkpoint
- GPT2-medium: [gdrive](https://drive.google.com/file/d/1y7RYsqrGt7njagHAmGrlA2a6jseGwkGX/view?usp=drive_link)
- GPT-baby:



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
python train_adamw.py
```

## 4. Citation



