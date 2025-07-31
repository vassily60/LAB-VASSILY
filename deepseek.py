# deepseek_nvtx_profile.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, record_function, ProfilerActivity
import torch.cuda.nvtx as nvtx

# Load DeepSeek model (adjust to exact one you want)
model_name = "deepseek-ai/deepseek-coder-1.3b-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="cuda"
)
model.eval()

# Sample input
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# NVTX-marked wrapper for forward
def forward_with_nvtx():
    nvtx.range_push("Forward Pass")
    with torch.no_grad():
        # Optional: add NVTX markers for attention layers
        def attention_marker_hook(module, input, output):
            nvtx.range_push(f"AttentionBlock: {module.__class__.__name__}")
            nvtx.range_pop()

        # Add NVTX to all attention layers
        for name, module in model.named_modules():
            if "attn" in name or "attention" in name.lower():
                module.register_forward_hook(attention_marker_hook)

        outputs = model(**inputs)
    nvtx.range_pop()  # end "Forward Pass"

# Set up profiler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/deepseek_nvtx"),
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    for _ in range(3):
        forward_with_nvtx()
        prof.step()

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total"))
