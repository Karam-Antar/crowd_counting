import torch 
print(f'Autocast active: {torch.is_autocast_enabled('cuda')}, Dtype: {torch.get_autocast_gpu_dtype()}')