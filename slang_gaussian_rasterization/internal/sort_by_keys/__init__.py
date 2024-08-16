import os
import torch

root_path = os.path.dirname(__file__)

sort_by_keys_cub = torch.utils.cpp_extension.load(name="sort_by_keys", 
                                                  sources=[os.path.join(root_path, "sort_by_keys.cu")])
