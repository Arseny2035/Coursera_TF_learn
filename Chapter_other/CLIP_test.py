import subprocess

CUDA_version = [s for s in subprocess.check_output(["nvcc", "--version"]).decode("UTF-8").split(", ") if s.startswith("release")][0].split(" ")[-1]
print("CUDA version:", CUDA_version)

if CUDA_version == "10.0":
    torch_version_suffix = "+cu100"
elif CUDA_version == "10.1":
    torch_version_suffix = "+cu101"
elif CUDA_version == "10.2":
    torch_version_suffix = ""
else:
    torch_version_suffix = "+cu110"

import numpy as np
import torch

print("Torch version:", torch.__version__)

# MODELS = {
#     "ViT-B/32":       "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
# }

# MODELS = {
#     "ViT-B/32":       "G:\NeuronNetworks\CLIP\ViT-B-32\ViT-B-32.pt",
# }
#
# wget {MODELS["ViT-B/32"]} -O model.pt

# import wget
# url = 'abc.com/sales.csv'
# filename = wget.download(url)
# print(filename)

