# Optimize GPT2 model with knowledge distillation and quantization.

## 1. Optimize
### 1.1 Input 
- Target model: GPT-2 (medium) - 350M params ~ 1.3GB 
### 1.2 Target
- Quantization: 1.3GB -> 200MB
- Knowledge Distillation (KD) with teacher model: GPT-neo/ GPT-2 (large)
- Serving on Cloud (AWS/Google Cloud) with triton server
### 1.3 Evaluation metric:
- Perplexity

### 1.4 Dataset

## 2. Resources
### 2.1 Framework, library
- Python 3.9
- Pytorch (2.0): https://pytorch.org/docs/stable/index.html
- ONNX: https://github.com/onnx/onnx
- TensorRT: https://github.com/NVIDIA/TensorRT
- Huggingface: https://github.com/huggingface

### 2.2 Survey
#### 2.2.1 Quantization
  - [ ] Pytorch quantization \
        https://pytorch.org/docs/stable/quantization.html
  - [ ] Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference \
        https://arxiv.org/pdf/1712.05877.pdf
  - [ ] INTEGER QUANTIZATION FOR DEEP LEARNING INFERENCE: PRINCIPLES AND EMPIRICAL EVALUATION \
        https://arxiv.org/pdf/2004.09602.pdf
  - [ ] ...

#### 2.2.1 Knowledge Distillation
  - [ ] Distilling the Knowledge in a Neural Network \
        https://arxiv.org/pdf/1503.02531.pdf
  - [ ] Improving Knowledge Distillation via Regularizing Feature Norm and Direction \
        https://arxiv.org/pdf/2305.17007v1.pdf \
        https://github.com/wangyz1608/knowledge-distillation-via-nd
  - [ ] ...

### 2.3 Repository / implemence paper
- https://github.com/yoshitomo-matsubara/torchdistill

### 3. Citation
