import torch
import math
import time
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model, TaskType

batch_size = 4
grad_accum = 8            
block_size = 512
max_iters = 3000
eval_interval = 200
lr = 5e-5
min_lr = 5e-6
warmup_iters = 100
weight_decay = 0.01

device = "cuda" 


tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_local")
tokenizer.pad_token = tokenizer.eos_token


class DollyBPE(Dataset):
    def __init__(self, data_path, block_size):
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        s = idx * self.block_size
        x = self.data[s:s + self.block_size]
        y = self.data[s + 1:s + self.block_size + 1]
        return x, y

train_dataset = DollyBPE("train_chat.txt", block_size)
val_dataset = DollyBPE("val_chat.txt", block_size)

print(f"训练 token 数: {len(train_dataset) * block_size}")
print(f"验证 token 数: {len(val_dataset) * block_size}")
print(f"训练 batch 数/epoch: {len(train_dataset) // batch_size}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)


base = GPT2LMHeadModel.from_pretrained("./gpt2_local").to(device)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base, lora_config)
model.print_trainable_parameters()


optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

def get_lr(it):
    if it < warmup_iters:
        return lr * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_val_loss():
    model.eval()
    losses = []
    for X, _ in val_loader:                
        X = X.to(device)
        outputs = model(input_ids=X, labels=X)
        losses.append(outputs.loss.item())
    model.train()
    return sum(losses) / len(losses)


best_val_loss = float("inf")
train_iter = iter(train_loader)
loss_ema = None
t0 = time.time()

for step in range(max_iters):
    lr_now = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr_now

    if step % eval_interval == 0 or step == max_iters - 1:
        val_loss = estimate_val_loss()
        print(f"步数 {step}: 验证集 Loss {val_loss:.4f}, lr {lr_now:.2e}", flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained("lora_best")
            tokenizer.save_pretrained("lora_best")
            print(f"✅ LoRA 适配器已保存 (Best Loss: {best_val_loss:.4f})")

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0

    for _ in range(grad_accum):
        try:
            xb, _ = next(train_iter)       
        except StopIteration:
            train_iter = iter(train_loader)
            xb, _ = next(train_iter)

        xb = xb.to(device)
        outputs = model(input_ids=xb, labels=xb)  
        loss = outputs.loss / grad_accum
        loss.backward()
        accum_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    loss_ema = accum_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * accum_loss
    if step % 50 == 0:
        dt = time.time() - t0
        t0 = time.time()
        print(f"step {step} train_loss_ema {loss_ema:.4f} ({dt:.1f}s/50step)", flush=True)

print("训练结束, best val loss:", best_val_loss)