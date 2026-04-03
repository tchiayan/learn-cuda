import cutlass 
import cutlass.cute as cute 
from cutlass.cute.runtime import from_dlpack
import torch 

@cute.kernel
def kernel_gemm(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    M = 2
    N = 2
    K = 2
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    block_dim_x, block_dim_y, _ = cute.arch.block_dim()
    
    x = bidx * block_dim_x + tidx
    y = bidy * block_dim_y + tidy 
    
    # let us print the memory access pattern for C[0, 0],assuming column major layout by default.
    if x == 0 and y == 0: # this is example of wrap divergence, only one thread will execute this branch
        for i in range(K): # default is column major, unless u specify dlpack with row major
            cute.printf(f"A[0]: {A[0]}") # Expecting value 1
            cute.printf(f"A[1]: {A[1]}") # Expecting value 3
            cute.printf(f"A[2]: {A[2]}") # Expecting value 2
            cute.printf(f"A[3]: {A[3]}") # Expecting value 4
    
    if x < M and y < N:
        temp = 0.0
        for i in range(K):
            temp += A[i*M + x] * B[y*K + i]
            if x == 0 and y == 0: # let just looking at C[0, 0] for simplicity
                # i=0, A = 1.0, i= 1, A=3
                # i=0, B = 5.0, i=1, B=6
                # 1.0 * 5.0 + 3.0 * 7.0
                cute.printf(f"Thread ({tidx}, {tidy}) computing C[{x}, {y}] | A[x*K+i]: {A[x*K + i]} | B[i*N+y]: {B[i*N + y]} ") 
        C[x*N + y] = temp
    

@cute.jit 
def gemm(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    kernel_gemm(A, B, C).launch(
        grid=[1, 1, 1], 
        block=[2, 2, 1]
    )
    
# Create input tensors
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=torch.float32)
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda", dtype=torch.float32)
C = torch.zeros(2, 2, device="cuda", dtype=torch.float32)

# A_dl = from_dlpack(A)
# B_dl = from_dlpack(B)
# C_dl = from_dlpack(C)

# A_tensor = cute.make_tensor(A_dl, shape=(2,2), stride=(2,1))
# B_tensor = cute.make_tensor(B_dl, shape=(2,2), stride=(2,1))
# C_tensor = cute.make_tensor(C_dl, shape=(2,2), stride=(2,1))
A_tensor = from_dlpack(A)
B_tensor = from_dlpack(B)
C_tensor = from_dlpack(C)


compiled = cute.compile(gemm, 
                        A_tensor, 
                        B_tensor, 
                        C_tensor)
ref_c = A @ B
compiled(A_tensor, B_tensor, C_tensor)
print(f"Reference : {ref_c}")
print(f"Result : {C}")
torch.testing.assert_close(C, ref_c, rtol=1e-3, atol=1e-3)