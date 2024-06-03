from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch


access_token = "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"
model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model, token=access_token)

model = AutoModelForCausalLM.from_pretrained(
    model, 
    token=access_token
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)