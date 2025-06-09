import torch
import triton
import triton.language as tl

# See GPU specifications:
# -----------------------
DEVICE = triton.runtime.driver.active.get_active_torch_device()  # device(type='cuda', index=0)
# DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')  # device(type='cuda', index=0)
triton.runtime.driver.active.get_current_target().backend  # 'cuda'
triton.runtime.driver.active.get_current_target().arch  # 120  --> on RTX 5060 Ti




# ...