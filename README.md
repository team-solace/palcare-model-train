<div align="center">


# ⚡ LitGPT

**Load, finetune, pretrain, evaluate, and deploy 20+ LLMs on your own data**

Uses the latest state-of-the-art techniques:

<pre>
✅ Scratch implementations  ✅ flash attention  ✅ fp4/8/16/32        ✅ LoRA, QLoRA, Adapter
✅ No abstractions          ✅ FSDP             ✅ 1-1000+ GPUs/TPUs  ✅ 20+ LLMs            
</pre>


---


![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)
![cpu-tests](https://github.com/lightning-AI/lit-stablelm/actions/workflows/cpu-tests.yml/badge.svg) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lit-stablelm/blob/master/LICENSE) [![Discord](https://img.shields.io/discord/1077906959069626439)](https://discord.gg/VptPCZkGNa)

<p align="center">
  <a href="https://lightning.ai/">Lightning AI</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#choose-from-20-llms">Models</a> •
  <a href="#finetune-an-llm">Finetune/pretrain</a> • 
  <a href="#deploy-an-llm">Deploy</a> •    
    <a href="#evaluate-an-llm">Evaluate</a> •
  <a href="#state-of-the-art-features">Features</a> •
  <a href="#training-recipes">Training recipes (YAML)</a> •
    <a href="#tutorials">Tutorials</a>
</p>

&nbsp;
  
<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-quick-start">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/get-started-badge.svg" height="36px" alt="Get started"/>
</a>

&nbsp;

</div>

# Load, finetune, pretrain, deploy LLMs Lightning fast ⚡⚡
LitGPT is a library of **lightning-fast** large language model (LLMs) **implemented from scratch** (Apache 2.0) with **no abstractions**.   

We reimplemented all model architectures and training recipes from scratch for 4 reasons:

✅ Apache 2.0 compliance to enable unlimited enterprise use.    
✅ Easy debugging/hacking with no abstraction layers and single file implementations.    
✅ Optimized model architectures to maximize performance, reduce costs, and speed up training.    
✅ Highly-optimized [recipe configs](#training-recipes) we have tested at enterprise scale.    

In addition to a simple Python API, it offers a command-line tool designed to easily [finetune](#finetune-an-llm), [pretrain](#pretrain-an-llm), [evaluate](#use-an-llm), and [deploy](#deploy-an-llm) [20+ LLMs](#choose-from-20-llms) **on your own data**. It features highly-optimized [training recipes](#training-recipes) for the world's most powerful open-source large language models (LLMs).

&nbsp;

# Quick start
Install LitGPT
```
pip install 'litgpt[all]'
```

Load and use any of the [20+ LLMs](#choose-from-20-llms):   
```python
from litgpt import LLM

llm = LLM.load("microsoft/phi-2")
text = llm.generate("Correct the spelling: Every summer, the familly enjoys a trip to the mountains.")
print(text)
# Corrected Sentence: Every summer, the family enjoys a vacation to the mountains.       
```

[Explore the full Python API docs](tutorials/python-api.md).

&nbsp;
&nbsp;

✅ Optimized for fast inference    
✅ Quantization    
✅ Runs on low-memory GPUs    
✅ No layers of internal abstractions    
✅ Optimized for production scale   

&nbsp;

<details>
  <summary>Advanced install options</summary>

&nbsp;

Install from source:

```bash
git clone https://github.com/Lightning-AI/litgpt
cd litgpt
pip install -e '.[all]'
```
</details>

---

&nbsp;

# Choose from 20+ LLMs
LitGPT has 🤯 **custom, from-scratch implementations** of [20+ LLMs](tutorials/download_model_weights.md) without layers of abstraction:

| Model | Model size | Author | Reference |
|----|----|----|----|
| Llama 3 | 8B, 70B | Meta AI | [Meta AI 2024](https://github.com/meta-llama/llama3) |
| Llama 2 | 7B, 13B, 70B | Meta AI | [Touvron et al. 2023](https://arxiv.org/abs/2307.09288) |
| Code Llama | 7B, 13B, 34B, 70B | Meta AI | [Rozière et al. 2023](https://arxiv.org/abs/2308.12950) |
| Mixtral MoE | 8x7B | Mistral AI | [Mistral AI 2023](https://mistral.ai/news/mixtral-of-experts/)                                                                      |
| Mistral | 7B | Mistral AI | [Mistral AI 2023](https://mistral.ai/news/announcing-mistral-7b/)                                                                         |
| CodeGemma | 7B | Google | [Google Team, Google Deepmind](https://ai.google.dev/gemma/docs/codegemma) |
| ... | ... | ... | ...   |

<details>
  <summary>See full list of 20+ LLMs</summary>

&nbsp;

#### All models

| Model | Model size | Author | Reference |
|----|----|----|----|
| CodeGemma | 7B | Google | [Google Team, Google Deepmind](https://ai.google.dev/gemma/docs/codegemma)                                                                 |
| Code Llama | 7B, 13B, 34B, 70B | Meta AI | [Rozière et al. 2023](https://arxiv.org/abs/2308.12950)                                                                   |
| Danube2 | 1.8B | H2O.ai | [H2O.ai](https://h2o.ai/platform/danube-1-8b/)                                                                                             |
| Dolly | 3B, 7B, 12B | Databricks | [Conover et al. 2023](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)      |
| Falcon | 7B, 40B, 180B | TII UAE | [TII 2023](https://falconllm.tii.ae)                                                                                              |
| FreeWilly2 (Stable Beluga 2) | 70B | Stability AI | [Stability AI 2023](https://stability.ai/blog/stable-beluga-large-instruction-fine-tuned-models)                 |
| Function Calling Llama 2 | 7B | Trelis | [Trelis et al. 2023](https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2)                                  |
| Gemma | 2B, 7B | Google | [Google Team, Google Deepmind](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)                                       |
| Llama 2 | 7B, 13B, 70B | Meta AI | [Touvron et al. 2023](https://arxiv.org/abs/2307.09288)                                                                           |
| Llama 3 | 8B, 70B | Meta AI | [Meta AI 2024](https://github.com/meta-llama/llama3)                                                                                   |
| LongChat | 7B, 13B | LMSYS | [LongChat Team 2023](https://lmsys.org/blog/2023-06-29-longchat/)                                                                       |
| MicroLlama | 300M | Ken Wang | [MicroLlama repo](https://github.com/keeeeenw/MicroLlama)
| Mixtral MoE | 8x7B | Mistral AI | [Mistral AI 2023](https://mistral.ai/news/mixtral-of-experts/)                                                                     |
| Mistral | 7B | Mistral AI | [Mistral AI 2023](https://mistral.ai/news/announcing-mistral-7b/)                                                                        |
| Nous-Hermes | 7B, 13B, 70B | NousResearch | [Org page](https://huggingface.co/NousResearch)                                                                          |
| OpenLLaMA | 3B, 7B, 13B | OpenLM Research | [Geng & Liu 2023](https://github.com/openlm-research/open_llama)                                                         |
| Phi | 1.3B, 2.7B | Microsoft Research  | [Li et al. 2023](https://arxiv.org/abs/2309.05463)                                                                          |
| Platypus | 7B, 13B, 70B |  Lee et al. | [Lee, Hunter, and Ruiz 2023](https://arxiv.org/abs/2308.07317)                                                               |
| Pythia | {14,31,70,160,410}M, {1,1.4,2.8,6.9,12}B | EleutherAI | [Biderman et al. 2023](https://arxiv.org/abs/2304.01373)                                            |
| RedPajama-INCITE | 3B, 7B | Together | [Together 2023](https://together.ai/blog/redpajama-models-v1)                                                                 |
| StableCode | 3B | Stability AI | [Stability AI 2023](https://stability.ai/blog/stablecode-llm-generative-ai-coding)                                                  |
| StableLM  | 3B, 7B | Stability AI | [Stability AI 2023](https://github.com/Stability-AI/StableLM)                                                                    |
| StableLM Zephyr | 3B | Stability AI | [Stability AI 2023](https://stability.ai/blog/stablecode-llm-generative-ai-coding)                                             |
| TinyLlama | 1.1B | Zhang et al. | [Zhang et al. 2023](https://github.com/jzhang38/TinyLlama)                                                                         |
| Vicuna | 7B, 13B, 33B | LMSYS | [Li et al. 2023](https://lmsys.org/blog/2023-03-30-vicuna/)                                                                          |

**Tip**: You can list all available models by running the `litgpt download list` command.


</details>

---

&nbsp;

# Advanced workflows
Use the command line interface to run advanced workflows such as pretraining or finetuning on your own data.   

## All commands   
After installing LitGPT, select the model and action you want to take on that model (finetune, pretrain, evaluate, deploy, etc...):

```bash
# ligpt [action] [model]
litgpt  download  meta-llama/Meta-Llama-3-8B-Instruct
litgpt  chat      meta-llama/Meta-Llama-3-8B-Instruct
litgpt  evaluate  meta-llama/Meta-Llama-3-8B-Instruct
litgpt  finetune  meta-llama/Meta-Llama-3-8B-Instruct
litgpt  pretrain  meta-llama/Meta-Llama-3-8B-Instruct
litgpt  serve     meta-llama/Meta-Llama-3-8B-Instruct
```

&nbsp;

### Finetune an LLM
[Finetune](tutorials/finetune.md) a model to specialize it on your own custom dataset:

<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-finetune">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>

&nbsp;

```bash
# 1) Download a pretrained model
litgpt download microsoft/phi-2

# 2) Finetune the model
curl -L https://huggingface.co/datasets/ksaw008/finance_alpaca/resolve/main/finance_alpaca.json -o my_custom_dataset.json

litgpt finetune microsoft/phi-2 \
  --data JSON \
  --data.json_path my_custom_dataset.json \
  --data.val_split_fraction 0.1 \
  --out_dir out/custom-model

# 3) Chat with the model
litgpt chat out/custom-model/final
```

&nbsp;

### Pretrain an LLM

[Train an LLM from scratch](tutorials/pretrain.md) on your own data via pretraining:

<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-pretrain">
<img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg"; alt="Open In Studio"/>
</a>

&nbsp;

```bash
mkdir -p custom_texts
curl https://www.gutenberg.org/cache/epub/24440/pg24440.txt --output custom_texts/book1.txt
curl https://www.gutenberg.org/cache/epub/26393/pg26393.txt --output custom_texts/book2.txt

# 1) Download a tokenizer
litgpt download EleutherAI/pythia-160m \
  --tokenizer_only True

# 2) Pretrain the model
litgpt pretrain EleutherAI/pythia-160m \
  --tokenizer_dir EleutherAI/pythia-160m \
  --data TextFiles \
  --data.train_data_path "custom_texts/" \
  --train.max_tokens 10_000_000 \
  --out_dir out/custom-model

# 3) Chat with the model
litgpt chat out/custom-model/final
```

&nbsp;

### Continue pretraining an LLM

[Continued pretraining](tutorials/pretrain.md#continued-pretraining-on-custom-data) is another way of finetuning that specializes an already pretrained model by training on custom data:


<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-continue-pretraining">
<img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg"; alt="Open In Studio"/>
</a>

&nbsp;

```bash
mkdir -p custom_texts
curl https://www.gutenberg.org/cache/epub/24440/pg24440.txt --output custom_texts/book1.txt
curl https://www.gutenberg.org/cache/epub/26393/pg26393.txt --output custom_texts/book2.txt

# 1) Download a pretrained model
litgpt download EleutherAI/pythia-160m

# 2) Continue pretraining the model
litgpt pretrain EleutherAI/pythia-160m \
  --tokenizer_dir EleutherAI/pythia-160m \
  --initial_checkpoint_dir EleutherAI/pythia-160m \
  --data TextFiles \
  --data.train_data_path "custom_texts/" \
  --train.max_tokens 10_000_000 \
  --out_dir out/custom-model

# 3) Chat with the model
litgpt chat out/custom-model/final
```

&nbsp;

### Evaluate an LLM

If you want to [evaluate](tutorials/evaluation.md) a downloaded, finetuned, or pretrained LLM on popular benchmark tasks, such as MMLU and Truthful QA, run the following command:

```bash
litgpt evaluate microsoft/phi-2 --tasks 'truthfulqa_mc2,mmlu'
```

[Read the full evaluation docs](tutorials/evaluation.md).

&nbsp;

### Deploy an LLM
Once you're ready to [deploy](tutorials/deploy.md) a finetuned LLM, run this command:

<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-serve">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>

&nbsp;

```bash
# locate the checkpoint to your finetuned or pretrained model and call the `serve` command:
litgpt serve microsoft/phi-2

# Alternative: if you haven't finetuned, download any checkpoint to deploy it:
litgpt download microsoft/phi-2
litgpt serve microsoft/phi-2
```

Test the server in a separate terminal and integrate the model API into your AI product:
```python
# 3) Use the server (in a separate Python session)
import requests, json
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"prompt": "Fix typos in the following sentence: Exampel input"}
)
print(response.json()["output"])
```

[Read the full deploy docs](tutorials/deploy.md).

&nbsp;

----

###  Use an LLM for inference

Use LLMs for [inference](tutorials/deploy.md) to test its chatting capabilities, run evaluations, or extract embeddings, etc.
Here's an example showing how to use the Phi-2 LLM.

<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-chat">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>

&nbsp;

```bash
# 1) List all available models in litgpt
litgpt download list

# 2) Download a pretrained model
litgpt download microsoft/phi-2

# 3) Chat with the model
litgpt chat microsoft/phi-2

>> Prompt: What do Llamas eat?
```

The download of certain models requires an additional access token. You can read more about this in the [download](tutorials/download_model_weights.md#specific-models-and-access-tokens) documentation. 
For more information on the different inference options, refer to the [inference](tutorials/inference.md) tutorial.

----
&nbsp;

# State-of-the-art features

✅ &nbsp;State-of-the-art optimizations: Flash Attention v2, multi-GPU support via fully-sharded data parallelism, [optional CPU offloading](tutorials/oom.md#do-sharding-across-multiple-gpus), and [TPU and XLA support](extensions/xla).

✅ &nbsp;[Pretrain](tutorials/pretrain.md), [finetune](tutorials/finetune.md), and [deploy](tutorials/inference.md)

✅ &nbsp;Reduce compute requirements with low-precision settings: FP16, BF16, and FP16/FP32 mixed.

✅ &nbsp;Lower memory requirements with [quantization](tutorials/quantize.md): 4-bit floats, 8-bit integers, and double quantization.

✅ &nbsp;[Configuration files](config_hub) for great out-of-the-box performance.

✅ &nbsp;Parameter-efficient finetuning: [LoRA](tutorials/finetune_lora.md), [QLoRA](tutorials/finetune_lora.md), [Adapter](tutorials/finetune_adapter.md), and [Adapter v2](tutorials/finetune_adapter.md).

✅ &nbsp;[Exporting](tutorials/convert_lit_models.md) to other popular model weight formats.

✅ &nbsp;Many popular datasets for [pretraining](tutorials/pretrain.md) and [finetuning](tutorials/prepare_dataset.md), and [support for custom datasets](tutorials/prepare_dataset.md#preparing-custom-datasets-for-instruction-finetuning).

✅ &nbsp;Readable and easy-to-modify code to experiment with the latest research ideas.

&nbsp;

---

# Training recipes

LitGPT comes with validated recipes (YAML configs) to train models under different conditions.  We've generated these recipes based on the parameters we found to perform the best for different training conditions.

Browse all training recipes [here](config_hub).

### Example

```bash
litgpt finetune \
  --config https://raw.githubusercontent.com/Lightning-AI/litgpt/main/config_hub/finetune/llama-2-7b/lora.yaml
```
<details>
  <summary>✅ Use configs to customize training</summary>
  
Configs let you customize training for all granular parameters like:

```yaml
# The path to the base model's checkpoint directory to load for finetuning. (type: <class 'Path'>, default: checkpoints/stabilityai/stablelm-base-alpha-3b)
checkpoint_dir: checkpoints/meta-llama/Llama-2-7b-hf

# Directory in which to save checkpoints and logs. (type: <class 'Path'>, default: out/lora)
out_dir: out/finetune/qlora-llama2-7b

# The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true". (type: Optional[str], default: null)
precision: bf16-true

...
```
</details>

<details>
  <summary>✅ Example: LoRA finetuning config</summary>

&nbsp;

```yaml
# The path to the base model's checkpoint directory to load for finetuning. (type: <class 'Path'>, default: checkpoints/stabilityai/stablelm-base-alpha-3b)
checkpoint_dir: checkpoints/meta-llama/Llama-2-7b-hf

# Directory in which to save checkpoints and logs. (type: <class 'Path'>, default: out/lora)
out_dir: out/finetune/qlora-llama2-7b

# The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true". (type: Optional[str], default: null)
precision: bf16-true

# If set, quantize the model with this algorithm. See ``tutorials/quantize.md`` for more information. (type: Optional[Literal['nf4', 'nf4-dq', 'fp4', 'fp4-dq', 'int8-training']], default: null)
quantize: bnb.nf4

# How many devices/GPUs to use. (type: Union[int, str], default: 1)
devices: 1

# The LoRA rank. (type: int, default: 8)
lora_r: 32

# The LoRA alpha. (type: int, default: 16)
lora_alpha: 16

# The LoRA dropout value. (type: float, default: 0.05)
lora_dropout: 0.05

# Whether to apply LoRA to the query weights in attention. (type: bool, default: True)
lora_query: true

# Whether to apply LoRA to the key weights in attention. (type: bool, default: False)
lora_key: false

# Whether to apply LoRA to the value weights in attention. (type: bool, default: True)
lora_value: true

# Whether to apply LoRA to the output projection in the attention block. (type: bool, default: False)
lora_projection: false

# Whether to apply LoRA to the weights of the MLP in the attention block. (type: bool, default: False)
lora_mlp: false

# Whether to apply LoRA to output head in GPT. (type: bool, default: False)
lora_head: false

# Data-related arguments. If not provided, the default is ``litgpt.data.Alpaca``.
data:
  class_path: litgpt.data.Alpaca2k
  init_args:
    mask_prompt: false
    val_split_fraction: 0.05
    prompt_style: alpaca
    ignore_index: -100
    seed: 42
    num_workers: 4
    download_dir: data/alpaca2k

# Training-related arguments. See ``litgpt.args.TrainArgs`` for details
train:

  # Number of optimizer steps between saving checkpoints (type: Optional[int], default: 1000)
  save_interval: 200

  # Number of iterations between logging calls (type: int, default: 1)
  log_interval: 1

  # Number of samples between optimizer steps across data-parallel ranks (type: int, default: 128)
  global_batch_size: 8

  # Number of samples per data-parallel rank (type: int, default: 4)
  micro_batch_size: 2

  # Number of iterations with learning rate warmup active (type: int, default: 100)
  lr_warmup_steps: 10

  # Number of epochs to train on (type: Optional[int], default: 5)
  epochs: 4

  # Total number of tokens to train on (type: Optional[int], default: null)
  max_tokens:

  # Limits the number of optimizer steps to run (type: Optional[int], default: null)
  max_steps:

  # Limits the length of samples (type: Optional[int], default: null)
  max_seq_length: 512

  # Whether to tie the embedding weights with the language modeling head weights (type: Optional[bool], default: null)
  tie_embeddings:

  #   (type: float, default: 0.0003)
  learning_rate: 0.0002

  #   (type: float, default: 0.02)
  weight_decay: 0.0

  #   (type: float, default: 0.9)
  beta1: 0.9

  #   (type: float, default: 0.95)
  beta2: 0.95

  #   (type: Optional[float], default: null)
  max_norm:

  #   (type: float, default: 6e-05)
  min_lr: 6.0e-05

# Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details
eval:

  # Number of optimizer steps between evaluation calls (type: int, default: 100)
  interval: 100

  # Number of tokens to generate (type: Optional[int], default: 100)
  max_new_tokens: 100

  # Number of iterations (type: int, default: 100)
  max_iters: 100

# The name of the logger to send metrics to. (type: Literal['wandb', 'tensorboard', 'csv'], default: csv)
logger_name: csv

# The random seed to use for reproducibility. (type: int, default: 1337)
seed: 1337
```
</details>

<details>
  <summary>✅ Override any parameter in the CLI:</summary>

```bash
litgpt finetune \
  --config https://raw.githubusercontent.com/Lightning-AI/litgpt/main/config_hub/finetune/llama-2-7b/lora.yaml \
  --lora_r 4
```
</details>

&nbsp;

# Community

We welcome all individual contributors, regardless of their level of experience or hardware. Your contributions are valuable, and we are excited to see what you can accomplish in this collaborative and supportive environment.

- [Request a feature](https://github.com/Lightning-AI/litgpt/issues)    
- [Submit your first contribution](https://lightning.ai/pages/community/tutorial/how-to-contribute-to-litgpt/)    
- [Join our Discord](https://discord.gg/VptPCZkGNa)    

&nbsp;

# Tutorials   

🚀 [Get started](tutorials/0_to_litgpt.md)    
⚡️  [Finetuning, incl. LoRA, QLoRA, and Adapters](tutorials/finetune.md)    
🤖 [Pretraining](tutorials/pretrain.md)    
💬 [Model evaluation](tutorials/evaluation.md)    
📘 [Supported and custom datasets](tutorials/prepare_dataset.md)    
🧹 [Quantization](tutorials/quantize.md)    
🤯 [Tips for dealing with out-of-memory (OOM) errors](tutorials/oom.md)   
🧑🏽‍💻 [Using cloud TPUs](extensions/xla)

&nbsp;


## Projects using LitGPT

Check out the projects below that use and build on LitGPT. If you have a project you'd like to add to this section, please don't hesitate to open a pull request.

&nbsp;

📊 **SAMBA: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling**

The [Samba](https://github.com/microsoft/Samba) project by researchers at Microsoft is built on top of the LitGPT code base and combines state space models with sliding window attention, which outperforms pure state space models.

&nbsp;

**🏆 NeurIPS 2023 Large Language Model Efficiency Challenge: 1 LLM + 1 GPU + 1 Day**

The LitGPT repository was the official starter kit for the [NeurIPS 2023 LLM Efficiency Challenge](https://llm-efficiency-challenge.github.io), which is a competition focused on finetuning an existing non-instruction tuned LLM for 24 hours on a single GPU.

&nbsp;

**🦙 TinyLlama: An Open-Source Small Language Model**

LitGPT powered the [TinyLlama project](https://github.com/jzhang38/TinyLlama) and [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385) research paper.

&nbsp;

**🍪 MicroLlama: MicroLlama-300M**

[MicroLlama](https://github.com/keeeeenw/MicroLlama) is a 300M Llama model pretrained on 50B tokens powered by TinyLlama and LitGPT.

&nbsp;

**🔬 Pre-training Small Base LMs with Fewer Tokens**

The research paper ["Pre-training Small Base LMs with Fewer Tokens"](https://arxiv.org/abs/2404.08634), which utilizes LitGPT, develops smaller base language models by inheriting a few transformer blocks from larger models and training on a tiny fraction of the data used by the larger models. It demonstrates that these smaller models can perform comparably to larger models despite using significantly less training data and resources.

&nbsp;

## Acknowledgements

This implementation extends on [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) and [nanoGPT](https://github.com/karpathy/nanoGPT), and it's **powered by [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡**.

- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@EleutherAI](https://github.com/EleutherAI) for [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) and the [Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [@Microsoft](https://github.com/microsoft) for [LoRA](https://github.com/microsoft/LoRA)
- [@tridao](https://github.com/tridao) for [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)

&nbsp;

## License

LitGPT is released under the [Apache 2.0](https://github.com/Lightning-AI/litgpt/blob/main/LICENSE) license.


## Citation

If you use LitGPT in your research, please cite the following work:

```bibtex
@misc{litgpt-2023,
  author       = {Lightning AI},
  title        = {LitGPT},
  howpublished = {\url{https://github.com/Lightning-AI/litgpt}},
  year         = {2023},
}
```

&nbsp;
