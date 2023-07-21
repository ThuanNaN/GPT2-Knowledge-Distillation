# Optimize GPT2 model with knowledge distillation and quantization.

## 1. Optimize
### 1.1 Input 
#### GPT2-medium
```
num_layer: 24
num_head: 16
emb: 1024
```

#### GPT2-baby-1
```
num_layer: 12
num_head: 8
emb: 512
```

#### GPT2-baby-2
```
num_layer: 6
num_head: 6
emb: 384
```


### 1.2 Evaluation metric:
- Perplexity - https://huggingface.co/docs/transformers/perplexity

### 1.3 Dataset
- Tiny Shakespeare - https://huggingface.co/datasets/tiny_shakespeare

### 1.4 Result
#### Teacher:
| Model        | Params | Size  |  loss  |
| :---         | :----: | :---: |  :---: |
| GPT2-medium  |  350M  | 1.3GB | 1.3469 |


#### Student:
| Model 	  | Params 	| Size   | loss    |        |
|:---------	  | :----:	| :---:  |:-------:|:------:|
|       	  |        	|        | Scratch | Distill|
| GPT-baby-1  |        	|        |         |        |
| GPT-baby-2  |  10M    |  10MB  | 1.4605  |        |


## 2. Quantazation



## 3. Citation



