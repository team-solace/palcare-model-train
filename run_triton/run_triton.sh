tensorrtllm_backend_dir=/home/ubuntu/tensorrtllm_backend
model_dir=/home/ubuntu/model
tokenizer_dir=/home/ubuntu/tokenizer
converted_model_dir=/home/ubuntu/model_engine
source_model_dir=/home/ubuntu/source_model
max_batch_size=256
max_context_len=8192

mkdir -p $tokenizer_dir

# Check if docker command runs
if ! [ -x "$(command -v docker)" ]; then
  # Add Docker's official GPG key:
  sudo apt-get update
  sudo apt-get install ca-certificates curl
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  # Add the repository to Apt sources:
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update -y

  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

# Git pull Tensorrt LLM
# Check if tensorrtllm_backend exists
if [ -d "tensorrtllm_backend" ]; then
  echo "tensorrtllm_backend exists"
else
  echo "cloning tensorrtllm_backend"
  git clone https://github.com/triton-inference-server/tensorrtllm_backend.git

  # Update the submodules
  cd "$tensorrtllm_backend_dir"
  git lfs install
  git submodule update --init --recursive
  cd ../
fi

# Install git lfs
if [ -x "$(command -v git-lfs)" ]; then
  echo "git-lfs exists"
else
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
  sudo apt-get install git-lfs
fi

if [ -d "TensorRT-LLM" ]; then
  echo "TensorRT-LLM exists"
else
  echo "cloning TensorRT-LLM"
  git submodule update --init --recursive
  git lfs install
  git lfs pull
  git clone https://github.com/NVIDIA/TensorRT-LLM.git
fi

# EDIT /etc/docker/daemon.json
# {
#     "runtimes": {
#         "nvidia": {
#             "path": "nvidia-container-runtime",
#             "runtimeArgs": []
#         }
#     }
# }

# Install nvidia toolkit
# sudo apt-get install -y nvidia-container-toolkit

# Run TRTLLM Docker
# docker run --rm --runtime=nvidia --gpus all --entrypoint /bin/bash -v "$converted_weights_dir":/converted_weights -v "$source_model_dir": source_model -it nvidia/cuda:12.4.0-devel-ubuntu22.04


# Check installation
python3 -c "import tensorrt_llm"
#####

# Install huggingface-cli
if [ -x "$(command -v huggingface-cli)" ]; then
  echo "huggingface-cli exists"
else
  pip install "huggingface_hub[cli]"
fi

# Download models
if [ -d "$source_model_dir" ]; then
  echo "source_model_dir exists"
else
  mkdir -p "$source_model_dir"
  huggingface-cli download lemousehunter/epflllm_meditron-7b-base --local-dir "$source_model_dir"
fi

if [ -d "$converted_model_dir" ]; then
  echo "converted_model_dir exists"
else
  mkdir -p $converted_model_dir
  huggingface-cli download lemousehunter/meditron-7b-medinstruct-aligned-trt --local-dir "$converted_model_dir"
fi

############ Cleaning Directories ############
rm -rf $model_dir

############ Setup Model ############
echo "copying model files"
mkdir -p "$model_dir"
cp -r "$PWD/tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm" "$model_dir/"
cp -r "$PWD/tensorrtllm_backend/all_models/inflight_batcher_llm/ensemble" "$model_dir/"
cp -r "$PWD/tensorrtllm_backend/all_models/inflight_batcher_llm/preprocessing" "$model_dir/"
cp -r "$PWD/tensorrtllm_backend/all_models/inflight_batcher_llm/postprocessing" "$model_dir/"
cp -r "$converted_model_dir/." "$model_dir/tensorrt_llm/1/"

echo "copying tokenizer"
cp "$source_model_dir/tokenizer.model" "$tokenizer_dir"
cp "$source_model_dir/tokenizer_config.json" "$tokenizer_dir"
cp "$source_model_dir/special_tokens_map.json" "$tokenizer_dir"
cp "$source_model_dir/added_tokens.json" "$tokenizer_dir/add_tokens.json"

echo "configuring tokenizer"
# Configuring preprocessing
python "$PWD/tensorrtllm_backend/tools/fill_template.py" --in_place \
    "$model_dir/preprocessing/config.pbtxt" \
     tokenizer_type:auto,tokenizer_dir:/tokenizer/,triton_max_batch_size:"$max_batch_size",add_special_tokens:true,preprocessing_instance_count:1

# Configuring postprocessig
python "$PWD/tensorrtllm_backend/tools/fill_template.py" --in_place \
    "$model_dir/postprocessing/config.pbtxt" \
     tokenizer_type:auto,tokenizer_dir:/tokenizer/,triton_max_batch_size:"$max_batch_size",add_special_tokens:true,postprocessing_instance_count:1

echo "configuring model"
# Configuring ensemble (chained model: preprocessing > tensorrt_llm > postprocessing)
python3 "$PWD/tensorrtllm_backend/tools/fill_template.py" --in_place \
     "$model_dir/ensemble/config.pbtxt" \
     triton_max_batch_size:"$max_batch_size"

# Configuring tensorrt_llm
python3 "$PWD/tensorrtllm_backend/tools/fill_template.py" --in_place \
     "$model_dir/tensorrt_llm/config.pbtxt" \
     triton_backend:tensorrtllm,decoupled_mode:false,engine_dir:/model/tensorrt_llm/1,triton_max_batch_size:"$max_batch_size",max_beam_width:1,max_attention_window_size:$max_context_len,kv_cache_free_gpu_mem_fraction:0.95,exclude_input_in_output:true,decoding_mode:top_k_top_p,enable_kv_cache_reuse:true,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:500,batch_scheduler_policy:max_utilization,enable_trt_overlap:true,enable_chunked_context:true

sudo docker run -p8000:8000 -p8001:8001 -p8002:8002 --rm -it --net host --shm-size=15g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v "$tensorrtllm_backend_dir":/tensorrtllm_backend -v "$model_dir":/model -v "$tokenizer_dir":/tokenizer nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3 bash
# sudo docker run -p8000:8000 -p8001:8001 -p8002:8002 --rm -it --net host --shm-size=15g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v /home/ubuntu/converted:/src_model -v /home/ubuntu/TensorRT-LLM:/TensorRT-LLM -v /home/ubuntu/tensorrtllm_backend_dir:/tensorrtllm_backend -v /home/ubuntu/CONVERTED:/weights -v /home/ubuntu/model_engine:/converted nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3 bash
# nvcr.io/nvidia/tritonserver:24.06-trtllm-python-py3 bash