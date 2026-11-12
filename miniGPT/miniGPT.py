import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
import tiktoken

batch_size=32
block_size=128
max_iters=10000
eval_interval=200
lr=3e-4
n_embd=256
n_head=8
n_layer=12
dropout=0.1

class DollyBPE(Dataset):
    def __init__(self,data_path,block_size,enc):
        with open(data_path,'r',encoding='utf-8') as f:
            text=f.read()
        self.enc=enc
        self.vocab_size=self.enc.n_vocab
        self.data=torch.tensor(self.enc.encode(text),dtype=torch.long)
        self.block_size=block_size
    def __len__(self):
        return len(self.data)-self.block_size
    def __getitem__(self,idx):
        x=self.data[idx:idx+self.block_size]
        y=self.data[idx+1:idx+self.block_size+1]
        return x,y
enc=tiktoken.get_encoding("gpt2")
vocab_size=enc.n_vocab
train_dataset=DollyBPE("train_chat.txt",block_size,enc)
val_dataset=DollyBPE("val_chat.txt",block_size,enc)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

class head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        A=q@k.transpose(-2,-1)*(C**(-0.5))
        A=A.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        A=F.softmax(A,dim=-1)
        A=self.dropout(A)
        return A@v

class mulheadatt(nn.Module):
    def __init__(self,head_size,n_head):
        super().__init__()
        self.heads=nn.ModuleList([head(head_size) for _ in range(n_head)])
        self.proj=nn.Linear(n_embd,n_embd)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        out=torch.cat([i(x) for i in self.heads],dim=-1)
        return self.dropout(self.proj(out))

class feedback(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_embd,n_embd*4),nn.ReLU(),nn.Linear(n_embd*4,n_embd),nn.Dropout(dropout))
    def forward(self,x):
        return self.net(x)

class block(nn.Module):
    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size=n_embd//n_head
        self.sa=mulheadatt(head_size,n_head)
        self.ffwd=feedback(n_embd)
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)
    def forward(self,x):
        x=x+self.sa(self.ln1(x))
        x=x+self.ffwd(self.ln2(x))
        return x

class minigpt(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.pos_embedding_table=nn.Embedding(block_size,n_embd)
        self.blocks=nn.Sequential(*[block(n_embd,n_head=n_head) for _ in range (n_layer)])
        self.ln_f=nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)
    def forward(self,x,y=None):
        B,T=x.shape
        tok_emb=self.token_embedding_table(x)
        pos_emb=self.pos_embedding_table(torch.arange(T,device='cuda'))
        a=tok_emb+pos_emb
        a=self.blocks(a)
        a=self.ln_f(a)
        logits=self.lm_head(a)
        if y is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            y=y.view(B*T)
            loss=F.cross_entropy(logits,y)
        return logits,loss
    def generate(self,x,max_new_tokens,temperature=1,top_k=None):
        for _ in range(max_new_tokens):
            x_cond=x[:,-block_size:]
            logits,loss=self(x_cond)
            logits=logits[:,-1,:]/temperature
            if top_k is not None:
                v,_=torch.topk(logits,min(top_k,vocab_size))
                logits[logits<v[:,[-1]]]=-float('inf')
            prob=F.softmax(logits,dim=-1)
            x_next=torch.multinomial(prob,num_samples=1)
            x=torch.cat((x,x_next),dim=1)
            current_text = enc.decode(x[0].tolist())
            if "<|End|>" in current_text:
                break
        return x

model=minigpt().to('cuda')
optimizer=torch.optim.AdamW(model.parameters(),lr=lr)

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split,loader in [('train',train_loader),('val',val_loader)]:
        losses=[]
        for i,(X,Y) in enumerate(loader):
            if i >=20:
                break
            X,Y=X.to('cuda'),Y.to('cuda')
            logits,loss=model(X,Y)
            losses.append(loss.item())
        out[split]=sum(losses)/len(losses)
    model.train()
    return out

best_val_loss=float('inf')
train_iter=iter(train_loader)
warmup_iters = 500
for step in range(max_iters):
    if step%eval_interval==0 or step==max_iters-1:
        losses=estimate_loss()
        print(f"步数 {step}: 训练集 Loss {losses['train']:.4f}, 验证集 Loss {losses['val']:.4f}")
        if losses['val']<best_val_loss:
            best_val_loss=losses['val']
            torch.save(model.state_dict(),'best_minigpt_model.pth')
            print("模型已保存 (Best: {best_val_loss:.4f})")
    try:
            xb,yb=next(train_iter)
    except StopIteration:
            train_iter=iter(train_loader)
            xb,yb=next(train_iter)
    xb,yb=xb.to('cuda'),yb.to('cuda')
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

model.load_state_dict(torch.load('best_minigpt_model.pth'))
model.eval()
print("\n"+"="*40)
while True:
    user_input=input("你:")
    if user_input.strip().lower() in ['exit','quit']:
        break
    if not user_input.strip():
        continue
    prompt=f"<|User|>: {user_input}\n<|Bot|>: "
    context=torch.tensor([enc.encode(prompt)],dtype=torch.long,device='cuda')
    generated_ids=model.generate(context,200,0.7,40)
    input_len=len(context[0])
    generate_text=generated_ids[0][input_len:].tolist()
    response = enc.decode(generate_text)
    if "<|End|>" in response:
        final_answer=response.split("<|End|>")[0]
    else:
        final_answer=response

    print(final_answer)





