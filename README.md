# Optimize GPT2 model with knowledge distillation and quantization.

## 1. Optimize
### 1.1 Input 
- Source model: GPT2-medium ~ 350M params ~ 1.3GB 
- Paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf 
- Checkpoint: https://huggingface.co/imthanhlv/vigpt2medium

### 1.2 Target
- Quantization / Prune: GPT2-m (1.3GB) -> GPT2-custom (200MB)
- Knowledge Distillation (KD) from teacher model: GPT2-m to GPT2-custom
- Triton server (https://github.com/triton-inference-server/server)
- Serving on Cloud (AWS/Google Cloud) with scaling and load balancing
### 1.3 Evaluation metric:
- Perplexity (https://huggingface.co/docs/transformers/perplexity)

### 1.4 Technical
- LoRA - https://arxiv.org/pdf/2106.09685.pdf
- QLoRA - https://arxiv.org/pdf/2305.14314.pdf
- Sophia - https://arxiv.org/pdf/2305.14342.pdf

## 2. Resources
### 2.1  Dataset
- Fashion dataset
### 2.3 Libraries (Python 3.9)
- Pytorch (2.0): https://pytorch.org/docs/stable/index.html
- PyTorch Accelerate: https://huggingface.co/docs/accelerate/index
- ONNX: https://github.com/onnx/onnx
- TensorRT: https://github.com/NVIDIA/TensorRT

### 2.4 Survey
#### Quantization / Prune
  - [ ] Pytorch quantization \
        https://pytorch.org/docs/stable/quantization.html
  - [ ] Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference \
        https://arxiv.org/pdf/1712.05877.pdf
  - [ ] INTEGER QUANTIZATION FOR DEEP LEARNING INFERENCE: PRINCIPLES AND EMPIRICAL EVALUATION \
        https://arxiv.org/pdf/2004.09602.pdf
  - [ ] DEEP COMPRESSION: COMPRESSING DEEP NEURAL NETWORKS WITH PRUNING, TRAINED QUANTIZATION AND HUFFMAN CODING \
        https://arxiv.org/pdf/1510.00149.pdf

#### Knowledge Distillation
  - [ ] Distilling the Knowledge in a Neural Network \
        https://arxiv.org/pdf/1503.02531.pdf
  - [ ] Improving Knowledge Distillation via Regularizing Feature Norm and Direction \
        https://arxiv.org/pdf/2305.17007v1.pdf \
        https://github.com/wangyz1608/knowledge-distillation-via-nd

### 2.5 Repository / implement paper
- https://github.com/yoshitomo-matsubara/torchdistill
- https://github.com/karpathy/nanoGPT
- https://github.com/Liuhong99/Sophia

### 3. Citation
