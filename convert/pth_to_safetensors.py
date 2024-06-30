import os

import torch
from transformers import AutoModel
from jsonargparse import CLI
from shutil import copyfile


def convert_model(pytorch_fpath: str, lit_model_path: str, base_model_hf_repo_id: str, output_fpath: str):
    print("processing", pytorch_fpath, "to", output_fpath)
    files_list = [f for f in os.listdir(lit_model_path) if os.path.isfile(os.path.join(lit_model_path, f)) and ".pth" not in f and ".gitattributes" not in f]
    print("=====================================")
    for f in files_list:
        print("copying", f, "to", output_fpath)
        copyfile(os.path.join(lit_model_path, f), os.path.join(output_fpath, f))
    print("=====================================")
    print("loading state dict from", pytorch_fpath)
    state_dict = torch.load(pytorch_fpath)
    print("=====================================")
    print("loading base model from", base_model_hf_repo_id)
    model = AutoModel.from_pretrained(base_model_hf_repo_id, state_dict=state_dict)
    print("=====================================")
    os.makedirs(output_fpath, exist_ok=True)
    print("saving to output_fpath", output_fpath)
    model.save_pretrained(output_fpath)


if __name__ == "__main__":
    CLI(convert_model)
