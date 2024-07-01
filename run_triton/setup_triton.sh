#!/bin/bash
############ Attribution ############
# Instructions are taken from https://towardsdatascience.com/deploying-llms-into-production-using-tensorrt-llm-ed36e620dac4

############ Option Checking ############
usage() {
    echo "Usage: setup.sh [--precision_type <precision>] [--converted_repo_id <converted_repo_id>] [--base_model_repo_id <base_model_repo_id>] [--docker_user_id <docker_user_id>] [--docker_image_name <docker_image_name>] [--replace_if_exists <replace_if_exists>] [--max_batch_size <max_batch_size>] [--max_input_len <max_input_len>] [--max_output_len <max_output_len>]" >&2
    echo "Supported precision types: 'float16', 'bfloat16, 'float32'" >&2
    echo "Supported replace_if_exists types: 'True', 'False" >&2
    echo "repo_id is the handle of the tensorrt model which you would('ve) pushed to huggingface" >&2
    echo "replace_if_exists is a boolean value that determines whether to re-push the model to huggingface if it already exists on hf hub" >&2
    echo "max_batch_size is optional, and will default to 1 if not specified"
    exit 1
}

# Initialize parameter variables with default values
precision=""
converted_repo_id=""
base_model_repo_id=""
docker_user_id=""
docker_image_name=""
replace_if_exists=""
max_batch_size=1
# TODO: add max_len as an argument
max_input_len=0
max_output_len=0
max_context_len=8192
max_beam_width=1
alpha=0.2
# max_num_tokens="$((max_len * 2 * 2))"

converted_weights_dir="$PWD/tmp/trt_engines/1-gpu/"
converted_model_dir="$PWD/tmp/trt_engines/compiled-model/"
tokenizer_dir="$PWD/tokenizers/llama/"
mkdir -p "$tokenizer_dir"

# Set options
options=$(getopt -o "" -l "precision_type:,converted_repo_id:,base_model_repo_id:,replace_if_exists:,docker_user_id:,docker_image_name:,max_batch_size:,max_input_len:,max_output_len:" -- "$@")
if [ $? -ne 0 ]; then
    echo "Invalid arguments."
    usage
    exit 1
fi
eval set -- "$options"

# Reads the named argument values
while [ $# -gt 0 ]; do
    case "$1" in
        --precision_type) precision="$2"; shift;;
        --converted_repo_id) converted_repo_id="$2"; shift;;
        --base_model_repo_id) base_model_repo_id="$2"; shift;;
        --docker_user_id) docker_user_id="$2"; shift;;
        --docker_image_name) docker_image_name="$2"; shift;;
        --replace_if_exists) replace_if_exists="$2"; shift;;
        --max_batch_size) max_batch_size="$2"; shift;;
        --max_input_len) max_input_len="$2"; shift;;
        --max_output_len) max_output_len="$2"; shift;;
        --) shift;;
    esac
    shift
done

# Check if converted_repo_id type is provided
if [ "$converted_repo_id" = "" ]; then
    echo "Error: --converted_repo_id option is required." >&2
    usage
    exit 1
fi

# Check if base_model_repo_id type is provided
if [ "$base_model_repo_id" = "" ]; then
    echo "Error: --base_model_repo_id option is required." >&2
    usage
    exit 1
fi

# Check if precision type is provided
if [ -z "$precision" ]; then
    echo "Error: --precision_type option is required." >&2
    usage
    exit 1
fi

# Check if replace_if_exists is provided
if [ -z "$replace_if_exists" ]; then
    echo "Error: --replace_if_exists option is required." >&2
    usage
    exit 1
fi

# Check if max_input_len is provided
if [ "$max_input_len" = 0 ]; then
    echo "Error: --max_input_len option is required." >&2
    usage
    exit 1
fi

# Check if max_output_len is provided
if [ "$max_output_len" = 0 ]; then
    echo "Error: --max_output_len option is required." >&2
    usage
    exit 1
fi

# Check if precision_type is in the list
allowed_types=("float16" "bfloat16" "float32")
case "${allowed_types[@]}" in
  *"$precision"*)
    ;;
  *)
    echo "Unknown precision type $precision."
    usage
    exit 1
    ;;
esac

# Check if replace_if_exists is in the list
allowed_types=("True" "False")
case "${allowed_types[@]}" in
  *"$replace_if_exists"*)
    ;;
  *)
    echo "Unknown replace_if_exists type $replace_if_exists."
    usage
    exit 1
    ;;
esac

source_model_dir="$PWD/tmp/hf_models/$base_model_repo_id"
model_dir="$PWD/models/$base_model_repo_id"

############ Import .env file ############
if [ ! -f .env ]; then
  echo "Error: .env file not found."
  exit 1
else
  echo "Importing .env file..."
  source .env set
fi

############ Install Build Tools ############
sudo apt install --assume-yes build-essential

############ Install Cuda ############
# check for cuda, install if not found
if ! [ -x "$(command -v nvidia-smi)" ] || ! [ -x "$(command -v nvcc)" ]; then
  echo "cuda not found, please install cuda manually following these steps:
  sudo -i
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
  sudo dpkg -i cuda-keyring_1.0-1_all.deb && \
  sudo apt-get update && \
  sudo apt-get -y install cuda-12-1
  exit
  "
  # exit 1
  sudo apt install nvidia-cuda-toolkit nvidia-cuda-toolkit-gcc

else
  echo "cuda found"
fi



############ Conda Environment Setup ############

if ! [ -x "$(command -v conda)" ]; then
  echo "conda not found, installing Conda..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh && \
  sh Miniconda3-py39_23.3.1-0-Linux-x86_64.sh -b
  source ~/miniconda3/bin/activate
else
  echo "conda found, skipping installation..."
fi

############ Initialize conda ############
source ~/miniconda3/etc/profile.d/conda.sh
conda init

############ Docker Setup ############
# Check if docker command works, if not install docker
if ! [ -x "$(command -v docker)" ]; then
  echo "docker not found, installing Docker..."
  # From https://docs.docker.com/engine/install/ubuntu/
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
  sudo apt-get update
else
  echo "docker found, skipping installation..."
fi

# Check docker push
if ! [ "$docker_user_id" = "" ]; then # if docker user id is provided then tags and pushes the docker image to docker hub
  if [ "$docker_image_name" = "" ]; then
    echo "Error: --docker_image_name option is required if docker_user_id is provided." >&2
    usage
    exit 1
  fi
fi

############ Import .env file ############
if [ ! -f .env ]; then
  echo "Error: .env file not found."
  exit 1
else
  echo "Importing .env file..."
  source .env
fi

############ Login to docker ############
echo "Logging into docker with user [$DOCKER_USER]"
REGISTRY_URL="https://index.docker.io/v1/"
sudo docker login $REGISTRY_URL --username="$DOCKER_USER" --password="$DOCKER_TOKEN"

############ Python packages installation ############
if ! { conda env list | grep 'tensor_rt_setup'; } >/dev/null 2>&1; then # checks for tensor_rt_setup environment, if not present then creates the environment and installs packages
  echo "'tensor_rt_setup' conda environment not found, creating environment and installing packages..."

  # create conda environment
  conda create -y -n tensor_rt_setup python=3.10
  conda activate tensor_rt_setup

  sudo apt --assume-yes install libopenmpi-dev # is required otherwise will face "error: Cannot compile MPI programs. Check your configuration!!!"
  # install brew
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
  brew install gcc
  brew install mpich
  brew link mpich # otherwise will lead to undefined reference errors

  # install python packages
  pip install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com
  pip install huggingface_hub pynvml
  conda install -y mpi4py
  pip install python-dotenv
  pip install -r TensorRT-LLM/examples/llama/requirements.txt
  pip install jsonargparse

else
  echo "tensor_rt_setup environment found, activating..."
  conda activate tensor_rt_setup
fi

############ Git Pull ############
if [ ! -d TensorRT-LLM ]; then # checks for TensorRT-LLm directory, if not present then clones the repo
  echo "cloning TensorRT-LLM..."
  git clone https://github.com/NVIDIA/TensorRT-LLM.git
fi

############ Cleanup and rebuild ############
echo "cleaning up model directories"
rm -rf "$converted_weights_dir"
rm -rf "$converted_model_dir"

echo "rebuilding model directories"
mkdir -p "$converted_weights_dir"
mkdir -p "$converted_model_dir"

############ Download Model ############
echo "downloading model..."
python download_model.py "$source_model_dir" "$base_model_repo_id" "$HF_WRITE_TOKEN"
echo "downloaded model"

############ Convert Weights ############
echo "converting weights"
# Documentation says we can use the llama example for compiling the model: https://docs.mistral.ai/self-deployment/trtllm/
python TensorRT-LLM/examples/llama/convert_checkpoint.py --model_dir "$source_model_dir" \
                                                         --output_dir "$converted_weights_dir" \
                                                         --dtype "$precision"
echo "weights converted"

############ Compile Model ############
echo "compiling model"

max_num_tokens_float="$(echo "scale=0; ($max_batch_size * $max_input_len * $alpha) + ($max_batch_size * $max_beam_width * (1 - $alpha))" | bc)"
max_num_tokens="$(printf "%.0f" "$max_num_tokens_float")"

echo "max_num_tokens: $max_num_tokens"

trtllm-build --checkpoint_dir "$converted_weights_dir" \
            --output_dir "$converted_model_dir" \
            --gemm_plugin "$precision" \
            --max_input_len "$max_input_len" \
            --max_output_len "$max_output_len" \
            --max_num_tokens  "$max_num_tokens" \
            --gpt_attention_plugin "$precision" \
            --remove_input_padding enable \
            --paged_kv_cache enable \
            --max_batch_size "$max_batch_size" \
            --use_fused_mlp \
            --context_fmha enable \
            --use_paged_context_fmha enable \
            --use_context_fmha_for_generation disable \


echo "model compiled"

############ Setup Model ############
echo "copying model files"
mkdir -p "$model_dir"
cp -r "$PWD/tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm" "$model_dir/"
cp -r "$PWD/tensorrtllm_backend/all_models/inflight_batcher_llm/ensemble" "$model_dir/"
cp -r "$PWD/tensorrtllm_backend/all_models/inflight_batcher_llm/preprocessing" "$model_dir/"
cp -r "$PWD/tensorrtllm_backend/all_models/inflight_batcher_llm/postprocessing" "$model_dir/"
cp -r "$converted_model_dir." "$model_dir/tensorrt_llm/1/"

echo "copying tokenizer"
cp "$source_model_dir/tokenizer.model" "$tokenizer_dir"
cp "$source_model_dir/tokenizer_config.json" "$tokenizer_dir"

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
     decoupled_mode:false,engine_dir:/model/tensorrt_llm/1,triton_max_batch_size:"$max_batch_size",max_beam_width:1,max_attention_window_size:$max_context_len,kv_cache_free_gpu_mem_fraction:0.95,exclude_input_in_output:true,decoding_mode:top_k_top_p,enable_kv_cache_reuse:true,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:500,batch_scheduler_policy:max_utilization,enable_trt_overlap:true,enable_chunked_context:true

############ Upload Model ############
# echo "uploading model"
# python upload_model.py "$converted_model_dir" "$converted_repo_id" "$replace_if_exists"
# echo "model uploaded"

############ Setup Triton ############
echo "setting up Triton"

# pull git
echo "cloning tensorrtllm_backend..."
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git

# Update the submodules
cd "$tensorrtllm_backend"
git lfs install
git submodule update --init --recursive

# Use the Dockerfile to build the backend in a container
# For x86_64
echo "building docker image for triton server..."
DOCKER_BUILDKIT=1 docker build -t "$docker_image_name" -f dockerfile/Dockerfile.trt_llm_backend .

echo "pushing triton server docker image"
if ! [ "$docker_user_id" = "" ]; then # if docker user id is provided then tags and pushes the docker image to docker hub
  sudo docker tag "$docker_image_name" "$docker_user_id"/"$docker_image_name"
  # sudo docker push "$docker_user_id"/"$docker_image_name"
fi


echo "Completed TensorRT triton server setup"