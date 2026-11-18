import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

device = "cuda"

tokenizer = GPT2Tokenizer.from_pretrained("lora_best")
tokenizer.pad_token = tokenizer.eos_token

base = GPT2LMHeadModel.from_pretrained("./gpt2_local")
model = PeftModel.from_pretrained(base, "lora_best").to(device)
model.eval()

while True:
    user_input = input("Instruction: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        break
    if not user_input.strip():
        continue

    prompt = f"Instruction: {user_input}\nResponse: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if "End" in response:
        response = response.split("End")[0]
    print(response)