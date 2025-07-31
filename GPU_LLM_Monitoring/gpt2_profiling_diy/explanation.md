# GPT2 Distilled version monitoring


In this folder you have serveral component, everything is related to gpt2 model, distilled version.

The code with the model and inference test are hosted on the vm to run on gpu. I imported the output profiling files using ```scp```command. 

There is both profiling with pytorch and nsys. 
 
Pytorch profiling can output several things, operation wise with their percentage of use in the ```.txt``` file, and then a more detailed version with time stamps in the ```.json``` file. A detailed explanation of Pytorch profiling is explaine below. 



## PyTorch Profiling

In PyTorch, operations prefixed with `aten::` refer to low-level tensor operations from the ATen library. These are the building blocks used internally during your model's forward and backward passes

Below are the `aten::` operations used in your profiling report:


### Matrix and Linear Algebra

- **`aten::addmm`**  
  Performs a fused matrix multiplication followed by addition:  
  `out = beta * self + alpha * mat1 @ mat2`  
  Commonly used in fully connected (`nn.Linear`) layers.

- **`aten::linear`**  
  High-level linear layer operation. Internally uses `aten::addmm`.

### Attention and Normalization

- **`aten::layer_norm`**  
  Applies Layer Normalization over the last certain number of dimensions.

- **`aten::native_layer_norm`**  
  Optimized native implementation of LayerNorm. Used for better performance.

- **`aten::scaled_dot_product_attention`**  
  Computes attention scores using scaled dot product, a key component of Transformer models.

- **`aten::_scaled_dot_product_efficient_attention`**  
  More efficient version of scaled dot product attention.

- **`aten::_efficient_attention_forward`**  
  Optimized wrapper for computing the forward pass of attention layers.

### Tensor Manipulation

- **`aten::view`**  
  Reshapes a tensor without copying data. Similar to NumPy's `reshape`.

- **`aten::reshape`**  
  Reshapes a tensor to a new shape. More flexible than `view`.

- **`aten::narrow`**  
  Returns a narrowed (sliced) view of the input tensor along a given dimension.

- **`aten::transpose`**  
  Swaps two dimensions of a tensor. Often used before matrix multiplication.

- **`aten::split`**  
  Splits a tensor into a list of smaller tensors along a given dimension.

- **`aten::slice`**  
  Extracts a slice of a tensor along a specific dimension.

- **`aten::cat`**  
  Concatenates a sequence of tensors along a specified dimension.

- **`aten::index`**  
  Selects elements from a tensor using indices.

- **`aten::as_strided`**  
  Returns a view of the tensor with custom strides. Powerful but risky if misused.

### Arithmetic Operations

- **`aten::add`**  
  Performs element-wise addition between tensors.

- **`aten::cumsum`**  
  Returns the cumulative sum of elements along a given dimension.

- **`aten::tanh`**  
  Applies the hyperbolic tangent function element-wise.

- **`aten::pow`**  
  Raises each element of the input tensor to a specified power.

### Logical and Bitwise Operations

- **`aten::eq`**  
  Compares two tensors element-wise for equality.

- **`aten::any`**  
  Returns `True` if any element of the input is `True`.

- **`aten::bitwise_not`**  
  Computes the bitwise NOT of each element.

- **`aten::__or__`**  
  Element-wise logical OR operation (alias for `aten::bitwise_or`).

### Embeddings and Masking

- **`aten::embedding`**  
  Retrieves rows from a lookup table, usually used for word embeddings.

- **`aten::masked_fill_`**  
  Replaces elements of a tensor with a scalar value where a mask is `True`.

### Tensor Management

- **`aten::empty`**  
  Allocates a new tensor without initializing its values.

- **`aten::clone`**  
  Returns a copy of the tensor.

- **`aten::contiguous`**  
  Returns a contiguous tensor in memory (needed for reshaping sometimes).

- **`aten::copy_`**  
  Copies the values from one tensor into another.

- **`aten::to`**  
  Moves tensor to a different device or converts to a different dtype.

### Utility Operations

- **`aten::item`**  
  Converts a one-element tensor into a standard Python number.

- **`aten::_local_scalar_dense`**  
  Internal operation used to extract scalar values.

- **`aten::argmax`**  
  Returns the index of the maximum value of all elements in the input tensor.

### Kernel Launch and Execution

- **`cudaLaunchKernel`**  
  Launches a kernel on the GPU. This is one of the most time-consuming operations, as it triggers GPU computation.

- **`cudaFuncGetAttributes`**  
  Retrieves attributes of a kernel function, such as shared memory usage, register count, etc., used for optimizing kernel launches.

- **`cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`**  
  Calculates the optimal number of active blocks per multiprocessor for a given kernel, helping determine the best occupancy for performance.

### Event and Stream Control

- **`cudaEventRecord`**  
  Records an event in a stream. Often used for timing and profiling GPU execution.

- **`cudaStreamSynchronize`**  
  Waits for all operations in a stream to complete. This is a blocking call and can impact performance if overused.

- **`cudaDeviceSynchronize`**  
  Waits for all device activity to complete. Similar to `cudaStreamSynchronize`, but global across all streams.

- **`cudaStreamIsCapturing`**  
  Checks whether a stream is currently capturing a CUDA graph.


### Memory Management

- **`cudaMalloc`**  
  Allocates device memory.

- **`cudaFree`**  
  Frees previously allocated device memory.

- **`cudaHostAlloc`**  
  Allocates pinned host memory that is page-locked for faster GPU transfers.

- **`cudaMemcpyAsync`**  
  Asynchronously copies memory between host and device, or between devices.

- **`cudaMemsetAsync`**  
  Asynchronously sets memory on the GPU to a given value.


### Device Queries and Utilities

- **`cudaGetDeviceCount`**  
  Returns the number of available CUDA-capable devices.

- **`cudaDeviceGetAttribute`**  
  Queries a device attribute such as max threads per block, warp size, etc.

- **`cudaPeekAtLastError`**  
  Checks for errors without resetting the error state.

- **`cudaGetSymbolAddress`**  
  Retrieves the address of a device variable.

- **`cudaGetDriverEntryPoint`**  
  Looks up a CUDA driver function entry point (usually internal usage).


## Nsight profiling locally
I imported the output files profiled from the virtual machine and they have either a `.sqlite`extension or a nsys in their file name. They allow for more detailed information about kernel used and cuda api calls made with time stamps but it is hard to associate them with layers. In a layer many calls to cuda kernels are made but it is not easy to associate them with the specific layers epesically because of the asynchronous nature of gpu computations. 
It might be possible to infer layers based on synchornization calls as they are made in constant intervals. Also they make sure that all the computations are at the same state to move forward. To be determined. 

Some graphs of the profiling sections are located in graphs folder.



