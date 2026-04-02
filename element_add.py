import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
import cutlass.cute.testing as testing

@cute.kernel
def elementwise_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,  # Coordinate tensor for predication
    shape: cute.Shape,
    thr_layout: cute.Layout,
    val_layout: cute.Layout
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    
    # Slice for this thread block
    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]
    blkCrd = cC[blk_coord]
    
    # Create copy operations
    copy_atom_load = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), gA.element_type
    )
    copy_atom_store = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), gC.element_type
    )
    
    # Create tiled copies
    tiled_copy_A = cute.make_tiled_copy_tv(
        copy_atom_load, thr_layout, val_layout
    )
    tiled_copy_B = cute.make_tiled_copy_tv(
        copy_atom_load, thr_layout, val_layout
    )
    tiled_copy_C = cute.make_tiled_copy_tv(
        copy_atom_store, thr_layout, val_layout
    )
    
    # Get thread slices
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    thr_copy_C = tiled_copy_C.get_slice(tidx)
    
    # Partition tensors
    thrA = thr_copy_A.partition_S(blkA)
    thrB = thr_copy_B.partition_S(blkB)
    thrC = thr_copy_C.partition_D(blkC)
    
    # Allocate register fragments
    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)
    
    # Setup predication for bounds checking
    thrCrd = thr_copy_C.partition_S(blkCrd)
    frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
    for i in range(0, cute.size(frgPred), 1):
        val = cute.elem_less(thrCrd[i], shape)
        frgPred[i] = val
    
    # Load from global memory
    cute.copy(copy_atom_load, thrA, frgA, pred=frgPred)
    cute.copy(copy_atom_load, thrB, frgB, pred=frgPred)
    
    # Compute: element-wise addition
    result = frgA.load() + frgB.load()
    frgC.store(result)
    
    # Store to global memory
    cute.copy(copy_atom_store, frgC, thrC, pred=frgPred)

@cute.jit
def elementwise_add(mA, mB, mC):
    # Define layouts
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    
    # Tile tensors
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    
    # Create coordinate tensor for predication
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    
    # Launch kernel
    elementwise_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1]
    )

# Usage
M, N = 1024, 512
A = torch.randn(M, N, dtype=torch.float32, device='cuda')
B = torch.randn(M, N, dtype=torch.float32, device='cuda')
C = torch.zeros(M, N, dtype=torch.float32, device='cuda')

mA = from_dlpack(A).mark_layout_dynamic()
mB = from_dlpack(B).mark_layout_dynamic()
mC = from_dlpack(C).mark_layout_dynamic()

# Compile with options
compiled = cute.compile(
    elementwise_add, mA, mB, mC,
    options="--generate-line-info"
)

# Execute
compiled(mA, mB, mC)

# Verify
torch.testing.assert_close(C, A + B)
print("Success!")