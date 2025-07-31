import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, schedule, tensorboard_trace_handler
import time

# ========= CONFIG ==========
model_name = "gpt2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========= LOAD MODEL AND TOKENIZER ==========
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Handle pad token issue for GPT-2
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# ========= SETUP TENSORBOARD ==========
writer = SummaryWriter(log_dir="./runs/gpt2_monitor")

# ========= METADATA ==========
total_params = sum(p.numel() for p in model.parameters())
writer.add_text("Model Info", f"Model: {model_name}, Parameters: {total_params:,}")

# ========= HOOK FUNCTION TO MONITOR ACTIVATIONS ==========
global_step = 0
def make_hook(name):
    def hook(module, input, output):
        global global_step
        if isinstance(output, torch.Tensor):
            act_mean = output.mean().item()
            buffer_size = output.nelement() * output.element_size()
            writer.add_scalar(f"{name}/activation_mean", act_mean, global_step)
            writer.add_scalar(f"{name}/buffer_bytes", buffer_size, global_step)
    return hook

# Register hooks to decoder blocks
for name, module in model.named_modules():
    if "h." in name and isinstance(module, torch.nn.Module):  # gpt2 blocks are named h.0, h.1, ...
        module.register_forward_hook(make_hook(name))

# ========= INPUT DATA ==========
sentences = [
    "Once upon a time, there was a robot learning to speak.",
    "Profiling helps you optimize performance.",
    "TensorBoard is useful for visualizing training runs.",
    "GPT-2 is a generative language model developed by OpenAI."
]

# ========= START PROFILING ==========
with profile(
    activities=[torch.profiler.ProfilerActivity.CPU] + (
        [torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []
    ),
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./runs/gpt2_monitor"),
    record_shapes=True,
    with_stack=True
) as prof:
    for i, sentence in enumerate(sentences):
        global_step = i
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Track memory
        if device.type == "cuda":
            mem = torch.cuda.memory_allocated(device)
            writer.add_scalar("GPU/AllocatedMemory", mem, global_step)

        prof.step()
        time.sleep(0.1)

writer.close()
print("✅ Done. Run `tensorboard --logdir=./runs/gpt2_monitor` to view profiling + activations.")
