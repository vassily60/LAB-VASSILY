import torch
from transformers import BertModel, BertTokenizer
import time

time.sleep(60)

# Load pre-trained BERT
model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# Dummy input
inputs = tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt").to(device)

with torch.no_grad():
    output = model(**inputs)

print(output)
