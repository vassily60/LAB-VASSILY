# Nsight GUI Profiling

## Instructions
The `output` file is the compiled version of one forward pass of gpt2 trained on shakespear data. It is used to profile the gpu activity, the model only contains one layer of transformer for simplicity but can be expended if needed.
It was obtained running this command:

`nvcc profile_gpt2.cu -L/usr/lib/aarch64-linux-gnu -lnvidia-ml -o output -lcublasLt -lcublas`

The flags are used to include the libraries and extentions the cuda code needs. 

## Observations

We can clearly observe the kernel API calls and the execution time of each kernel. One thing to note is the rapid and close API kernel calls from the cpu. The cpu calls these kernels and it then the gpu manage a sort of queue to process each api call in order. A syncronization is required to make sure the cpu does not get too far ahead from the gpu processing the data. 


## Nsys Profiling

Below is a brief description of each kernel and CUDA runtime function that appeared in the Nsight Systems GUI.

### `elementwise_kernel`  
 
Executes a simple parallel element-wise operation across tensors. Often used for copying or transforming data one element at a time.  
Common during tensor preparation steps (e.g., casting, data copy).


### `cudaLaunchKernel`  

CUDA runtime function responsible for launching a kernel on the GPU. The correlation ID links this to the actual kernel execution.  
Always appears before the corresponding kernel run; useful for performance correlation.

### `cudaStreamIsCapturing`  

Checks whether a CUDA stream is currently capturing operations for a CUDA Graph.  
Appears when graph-based execution is employed to optimize performance.

### `fmha_cutlassF_f32_aligned_64x64_rf_sm75`  
Kernel used for Flash Attention (FMHA = Fused Multi-Head Attention) implemented with CUTLASS for NVIDIA SM75 architecture.  
Indicates the execution of the attention mechanism in transformer-based models.  

### `gemmSN_NN_kernel`   
A specialized GEMM (matrix multiply) kernel, likely used for feedforward layers or Q/K/V projection in attention.  
Appears during heavy matrix multiplication, a core operation in transformers.


### `vectorized_elementwise_kernel`  
Applies a vectorized (SIMD) addition operation element-wise across input tensors.  
Often used for combining residual connections or adding biases.

### `vectorized_layer_norm_kernel`  
Performs layer normalization across tensor dimensions in a vectorized manner.  
Used in almost every transformer block post-attention or feedforward layer.

**`correlation` fields and `cudaLaunchKernel` pairs**  
  Use correlation IDs to match kernel launches with execution blocks and infer when operations happen.

**`device synchronize` (not shown explicitly but implied)**  
Indicates a synchronization barrier where the host waits for GPU to finish execution.  
Can often signify the end of a layer or major operation in inference.


By tracing correlation IDs and the sequence of these kernels, you can roughly infer the stages of the LLM pipeline (e.g., token embedding → attention → FFN → normalization).








