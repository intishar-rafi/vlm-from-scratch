# Vision-Language Model from Scratch: PyTorch Implementation

A from-scratch PyTorch implementation of a Vision-Language Model (PaliGemma-style). No high-level VLM libraries, no black boxes.

This project re-implements every core component of the architecture by hand: a contrastive vision encoder (SigLIP), a decoder-only language model (Gemma) with grouped-query attention and rotary positional embeddings, the image-text fusion pipeline, and a KV-cached inference loop for efficient autoregressive generation.

## What This Demonstrates

- **Vision Transformer (SigLIP) from scratch**: patch embedding via convolution, learned positional embeddings, multi-head self-attention, and the contrastive (sigmoid loss) pretraining objective that produces the vision backbone.
- **Gemma decoder-only language model**: RMSNorm, grouped-query attention (with a from-scratch explanation and implementation of why GQA reduces GPU memory-bandwidth bottlenecks, not just parameter count), and rotary positional embeddings (RoPE).
- **Multimodal fusion**: linear projection of image patch embeddings into the language model's embedding space, and merging of image and text tokens into a single input sequence using placeholder-token replacement.
- **KV caching for inference**: a hand-implemented key/value cache supporting the prefill and incremental decode phases, avoiding redundant computation during autoregressive generation.
- **Custom attention masking logic**: PaliGemma's non-causal prefix / causal suffix masking strategy, implemented and explained.
- **Sampling strategies**: greedy decoding, top-p (nucleus) sampling, and temperature scaling.
- **Image and text preprocessing pipeline**: replicates PaliGemma's tokenizer/processor behavior, including image resizing/normalization and prompt construction with image placeholder tokens.

## Architecture Overview

```
Image → [SigLIP Vision Encoder] → patch embeddings
                                        │
                                        ▼
                            [Linear Projection]
                                        │
                                        ▼
Text prompt → [Gemma Tokenizer] → text embeddings
                                        │
                                        ▼
                [Merge image + text embeddings]
                                        │
                                        ▼
                    [Gemma Decoder (w/ KV Cache)]
                                        │
                                        ▼
                          Generated text output
```

## Pipeline

```
INPUT IMAGE (e.g. 224x224x3)
        │
        ▼
┌─────────────────────────────────────────────┐
│ SIGLIP VISION ENCODER                        │
│                                               │
│  1. Patchify: Conv2D, kernel=patch_size,     │
│     stride=patch_size                        │
│     → 224x224 image, 16x16 patches           │
│     → 196 patch embeddings                   │
│                                               │
│  2. Flatten + add learned positional         │
│     embeddings (one per patch position)      │
│                                               │
│  3. N x Transformer Encoder Layers:           │
│     LayerNorm → Multi-Head Self-Attention     │
│     → residual add                            │
│     LayerNorm → MLP (GELU) → residual add     │
│     (no causal mask - every patch attends     │
│     to every other patch)                     │
│                                               │
│  4. Final LayerNorm                          │
│                                               │
│  → output: 196 contextualized patch          │
│    embeddings, each of size hidden_dim       │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ MULTIMODAL PROJECTOR                         │
│  Linear layer: vision hidden_dim              │
│  → language model hidden_dim                  │
│  (so image and text embeddings share one     │
│  dimensionality and can sit in the same       │
│  sequence)                                    │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ TOKEN FUSION                                 │
│                                               │
│  Text prompt → Gemma tokenizer → input IDs    │
│  Prompt is pre-padded with N placeholder      │
│  <image> tokens (N = number of patches)       │
│                                               │
│  input IDs → embedding lookup → text embeds   │
│                                               │
│  Placeholder <image> token embeddings are     │
│  overwritten in-place with the projected      │
│  patch embeddings from the vision encoder     │
│                                               │
│  → single merged sequence:                    │
│    [image emb 1 ... image emb 196]            │
│    [BOS] [prompt tokens...] [\n]              │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ ATTENTION MASK CONSTRUCTION                  │
│  Image tokens + text prefix: unmasked          │
│  (bidirectional, no causality)                │
│  Only the generated suffix tokens get a        │
│  causal mask                                  │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ GEMMA DECODER (N x layers)                    │
│                                               │
│  Per layer:                                   │
│  RMSNorm                                      │
│    → Grouped-Query Self-Attention:            │
│       - Q/K/V linear projections               │
│       - split into heads (fewer KV heads       │
│         than Q heads, shared across groups)    │
│       - RoPE applied to Q and K                │
│         (rotates each head's vector by an      │
│         angle proportional to position)        │
│       - KV cache: read/write K,V for this      │
│         layer (prefill = write full prompt,     │
│         decode = append 1 new token)           │
│       - scaled dot-product attention            │
│         + attention mask                       │
│       - output projection                      │
│    → residual add                              │
│  RMSNorm                                      │
│    → MLP (gate/up/down projections, GELU)      │
│    → residual add                              │
│                                               │
│  → contextualized hidden states               │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ OUTPUT HEAD                                  │
│  Final RMSNorm                                │
│  → Linear (language modeling head, tied        │
│    weights with the input embedding layer)     │
│  → logits over vocabulary                      │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ SAMPLING                                     │
│  Take logits for the last position only        │
│  → temperature scaling → softmax               │
│  → top-p filtering (or greedy argmax)           │
│  → next token                                  │
│  → append to KV cache, repeat until EOS         │
│    or max_tokens                               │
└─────────────────────────────────────────────┘
        │
        ▼
   Generated text output
```

## Repository Structure

```
├── modeling_siglip.py       # SigLIP vision encoder (ViT + contrastive pretraining components)
├── modeling_gemma.py        # Gemma decoder-only LM: RMSNorm, GQA, RoPE, KV cache, PaliGemma fusion
├── processing_paligemma.py  # Image preprocessing + prompt/token construction
├── utils.py                 # Model/weight loading from Hugging Face checkpoints
├── inference.py             # End-to-end inference script (greedy / top-p sampling)
└── README.md
```

## Setup

```bash
git clone <this-repo-url>
cd vlm-from-scratch
pip install -r requirements.txt
```

Download the pretrained PaliGemma weights from [Hugging Face](https://huggingface.co/google/paligemma-3b-pt-224) (requires accepting the license) and place them locally.

## Usage

```bash
python inference.py \
    --model_path "<path-to-downloaded-weights>" \
    --prompt "this building is" \
    --image_file_path "test_images/example.jpg" \
    --max_tokens_to_generate 100 \
    --temperature 0.8 \
    --top_p 0.9 \
    --do_sample False \
    --only_cpu False
```

**Hardware note**: PaliGemma is a ~3B parameter model. CPU inference works but is slow; a free-tier GPU (e.g. Colab T4) makes it dramatically faster. Fine-tuning requires a GPU with meaningful VRAM (16GB+ with techniques like LoRA).

## Key Concepts Implemented

| Concept                                                               | Where                                                     |
| --------------------------------------------------------------------- | --------------------------------------------------------- |
| Contrastive vision-language pretraining (CLIP to SigLIP sigmoid loss) | `modeling_siglip.py`                                      |
| Patch embeddings via strided convolution                              | `modeling_siglip.py`                                      |
| Multi-head vs. grouped-query attention                                | `modeling_gemma.py`                                       |
| Rotary positional embeddings (RoPE)                                   | `modeling_gemma.py`                                       |
| RMSNorm vs. LayerNorm                                                 | `modeling_gemma.py`                                       |
| KV cache (prefill and decode)                                         | `modeling_gemma.py`                                       |
| Image/text token fusion                                               | `modeling_gemma.py` (`PaliGemmaForConditionalGeneration`) |
| Top-p / temperature sampling                                          | `inference.py`                                            |
