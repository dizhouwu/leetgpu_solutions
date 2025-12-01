#include <cuda_runtime.h>
#include <cstdio>

// --- Configuration Constants ---
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 1024
#define MAX_K 128 // Increased Max K to handle k=100 requests (128 for margin)

// A very small number to represent 'dead' or sentinel values
#define NEG_INF -1e30f
#define POS_INF 1e30f

// --- Helper Struct (Key/Value pair for Top-K) ---
struct KeyValuePair {
    float val;
    int idx;
};

// --- Value Comparison and Selection ---

// Selects the better candidate (largest value, or smallest index on tie)
__device__ __forceinline__ KeyValuePair select_topk(KeyValuePair a, KeyValuePair b) {
    if (a.val > b.val) return a;
    if (a.val < b.val) return b;
    // Tie-breaker: smaller index wins
    return (a.idx < b.idx) ? a : b;
}

// Sentinel value for the largest-value reduction
__device__ __forceinline__ float topk_dead_val_largest() {
    return NEG_INF;
}

// --- Warp Reduction ---
__device__ __forceinline__ KeyValuePair warp_reduce_topk(KeyValuePair val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        KeyValuePair other;
        // Broadcast fields individually (required for structs)
        other.val = __shfl_down_sync(0xffffffff, val.val, offset);
        other.idx = __shfl_down_sync(0xffffffff, val.idx, offset);
        val = select_topk(val, other);
    }
    return val;
}

// --- Block Reduction ---
__device__ __forceinline__ KeyValuePair block_reduce_topk(KeyValuePair val) {
    // 1. Warp-level reduction
    val = warp_reduce_topk(val);

    // FIX: If the block fits within a single warp, the reduction is complete.
    if (blockDim.x <= WARP_SIZE) {
        // Broadcast the final result to all threads in the active mask
        float final_val = __shfl_sync(0xffffffff, val.val, 0); 
        int final_idx = __shfl_sync(0xffffffff, val.idx, 0);  
        return {final_val, final_idx};
    }

    // 2. Inter-warp reduction via Shared Memory (only for blocks > WARP_SIZE)
    // Removed MAX_BLOCKS dependence for robustness
    __shared__ KeyValuePair s_warp_results[MAX_BLOCK_SIZE / WARP_SIZE];
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    if (threadIdx.x % WARP_SIZE == 0) {
        s_warp_results[warp_id] = val;
    }
    __syncthreads();

    // 3. Final reduction by the first warp
    if (warp_id == 0) {
        if (threadIdx.x < num_warps) {
            val = s_warp_results[threadIdx.x];
        } else {
            val.val = topk_dead_val_largest();
            val.idx = -1;
        }
        val = warp_reduce_topk(val);
    }

    // 4. Broadcast the final result to all threads
    // This uses the correct, field-by-field shfl_sync approach.
    float final_val = __shfl_sync(0xffffffff, val.val, 0); 
    int final_idx = __shfl_sync(0xffffffff, val.idx, 0);  

    // Reconstruct and return the KeyValuePair
    return {final_val, final_idx};
}

// --- Stage 1 Kernel (Local Reduction and Kill) ---
__global__ void topk_stage1_iterative(
    const float* __restrict__ in_buffer,
    float* __restrict__ local_topk_vals,
    int* __restrict__ local_topk_idxs,
    int num_elements, // Size of the last dimension
    int max_k
) {
    // Shared memory to hold the single best candidate found by each thread's iteration
    extern __shared__ KeyValuePair topk_sram[];
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int bid = blockIdx.x;

    // --- 1. Initial Scan and Shared Memory Write ---

    // Find the single best candidate across the entire input stripe for this thread
    KeyValuePair thread_best = {topk_dead_val_largest(), -1};
    int total_threads = gridDim.x * block_size;
    int start_idx = bid * block_size + tid; // Corrected stride access for simplicity

    for (int i = start_idx; i < num_elements; i += total_threads) {
        float current_val = in_buffer[i];
        if (current_val > thread_best.val) {
            thread_best.val = current_val;
            thread_best.idx = i;
        }
    }
    
    // Store each thread's single best candidate into shared memory
    topk_sram[tid] = thread_best;
    __syncthreads();


    // --- 2. Iterative Reduction and Kill (K times) ---

    for (int k_idx = 0; k_idx < max_k; ++k_idx) {
        // Read candidate from shared memory
        KeyValuePair partial = topk_sram[tid];

        // Perform block-level reduction to find the overall maximum (total)
        KeyValuePair total = block_reduce_topk(partial);

        // Thread 0 handles global memory write and 'kill' operation
        if (tid == 0) {
            // Store the local result (candidate for Stage 2)
            local_topk_vals[bid * max_k + k_idx] = total.val;
            local_topk_idxs[bid * max_k + k_idx] = total.idx;
            
            // Kill the found maximum in shared memory (only if a valid result was found)
            if (total.idx != -1) {
                // Find the contributing thread and kill the value in shared memory
                for (int i = 0; i < block_size; ++i) {
                    if (topk_sram[i].val == total.val && topk_sram[i].idx == total.idx) {
                        topk_sram[i].val = topk_dead_val_largest();
                        topk_sram[i].idx = -1;
                        break; // Only need to kill one instance
                    }
                }
            }
        }
        __syncthreads();
    }
}

// --- Stage 2 Kernel (Global Merge of Candidates) ---
// This runs on a single block (or small grid) to merge the results from Stage 1.
__global__ void topk_stage2_merge(
    const float* __restrict__ candidates_val,
    const int* __restrict__ candidates_idx,
    float* __restrict__ final_topk_vals,
    int* __restrict__ final_topk_idxs,
    int num_candidates, // total_candidates from Stage 1
    int k // final K
) {
    extern __shared__ KeyValuePair merge_sram[];
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // 1. Load candidates into Shared Memory
    if (tid < num_candidates) {
        merge_sram[tid] = {candidates_val[tid], candidates_idx[tid]};
    } else if (tid < block_size) {
        // Pad shared memory with dead values if block is larger than candidates
        merge_sram[tid] = {topk_dead_val_largest(), -1};
    }
    __syncthreads();
    
    // 2. Iterative Reduction and Kill (K times)
    for (int k_idx = 0; k_idx < k; ++k_idx) {
        // Perform reduction on the shared memory array
        KeyValuePair partial = merge_sram[tid];
        KeyValuePair total = block_reduce_topk(partial);

        // Thread 0 handles global memory write and 'kill' operation
        if (tid == 0) {
            // Write final result directly to the device output buffer
            final_topk_vals[k_idx] = total.val;
            final_topk_idxs[k_idx] = total.idx;
            
            // Kill the found maximum in shared memory (only if a valid result was found)
            if (total.idx != -1) {
                // Find the contributing thread and kill the value in shared memory
                for (int i = 0; i < num_candidates; ++i) {
                    if (merge_sram[i].val == total.val && merge_sram[i].idx == total.idx) {
                        merge_sram[i].val = topk_dead_val_largest();
                        merge_sram[i].idx = -1;
                        break; 
                    }
                }
            }
        }
        __syncthreads();
    }
}


// --- Host Wrapper ---

extern "C" void solve(const float* input, float* output, int N, int k) {
    if (N <= 0 || k <= 0 || k > N) return;
    k = (k > MAX_K) ? MAX_K : k;

    // --- Stage 1 Setup ---
    int max_merge_candidates = MAX_BLOCK_SIZE; // 1024
    
    // Calculate the maximum number of blocks that can be merged in Stage 2
    // This ensures total_candidates <= MAX_BLOCK_SIZE, allowing Stage 2 to run in one block.
    int max_allowed_blocks = (k > 0) ? (max_merge_candidates / k) : 1; 
    
    int blocks = N / BLOCK_SIZE;
    if (N % BLOCK_SIZE != 0 || blocks == 0) blocks++;
    
    // Cap blocks based on the merge capacity
    blocks = (blocks > max_allowed_blocks) ? max_allowed_blocks : blocks;
    blocks = (blocks == 0) ? 1 : blocks;

    // Device buffers to hold candidates from all blocks
    int total_candidates = blocks * k;
    float* d_block_tops_val = nullptr;
    int* d_block_tops_idx = nullptr;
    cudaMalloc(&d_block_tops_val, total_candidates * sizeof(float));
    cudaMalloc(&d_block_tops_idx, total_candidates * sizeof(int));
    
    int shmem_bytes_stage1 = BLOCK_SIZE * sizeof(KeyValuePair);

    // Launch Stage 1: Find local top-K candidates
    topk_stage1_iterative<<<blocks, BLOCK_SIZE, shmem_bytes_stage1>>>(
        input,
        d_block_tops_val,
        d_block_tops_idx,
        N,
        k
    );
    
    cudaDeviceSynchronize();

    // --- Stage 2 Setup ---
    // Enforce a minimum block size of WARP_SIZE (32) for robust reduction in Stage 2.
    int merge_block_size = WARP_SIZE;
    if (total_candidates > WARP_SIZE) {
        merge_block_size = (total_candidates < MAX_BLOCK_SIZE) ? total_candidates : MAX_BLOCK_SIZE;
    }

    int shmem_bytes_stage2 = merge_block_size * sizeof(KeyValuePair);

    // We need a device buffer for the final indices (since output only provides values)
    int* d_final_idx = nullptr;
    cudaMalloc(&d_final_idx, k * sizeof(int));

    // Launch Stage 2: Merge all candidates to find the final global top-K.
    // Writes the final values directly to the output device pointer.
    topk_stage2_merge<<<1, merge_block_size, shmem_bytes_stage2>>>(
        d_block_tops_val, 
        d_block_tops_idx,
        output, // Final values written here
        d_final_idx, // Final indices written here (and implicitly ignored by test runner)
        total_candidates,
        k
    );
    
    cudaDeviceSynchronize();
    
    // --- Cleanup ---

    cudaFree(d_block_tops_val);
    cudaFree(d_block_tops_idx);
    cudaFree(d_final_idx);
}
