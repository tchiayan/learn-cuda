"""
uv pip install torch 
uv pip install nvidia-cutlass-dsl
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# Define kernel using @cute.kernel decorator
@cute.kernel
def hello_world_kernel(tensor: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    
    idx = bidx * cute.arch.block_dim()[0] + tidx
    cute.printf("Hello from thread %d, block %d!, idx %d : %d\\n", tidx, bidx, idx, tensor[idx])
    tensor[idx] = tensor[idx] * 2

# Wrapper function with @cute.jit
@cute.jit
def hello_world(tensor):
    num_threads = 8
    num_blocks = (cute.size(tensor) + num_threads - 1) // num_threads
    
    hello_world_kernel(tensor).launch(
        grid=[num_blocks, 1, 1],
        block=[num_threads, 1, 1]
    )

# Create tensor and run
data = torch.range(0, 15, dtype=torch.int32, device='cuda').reshape(4,4)
print("Original data:", data)  # Should be all 1.0
tensor = from_dlpack(data)

# Compile and execute
compiled = cute.compile(hello_world, tensor)
compiled(tensor)

print(f"Result: {data}")  # Should be all 2.0