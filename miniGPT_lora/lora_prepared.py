from modelscope.hub.snapshot_download import snapshot_download

model_id='AI-ModelScope/gpt2'
local_dir='./gpt2_local'

snapshot_download(model_id, local_dir=local_dir)
print(f"模型已成功下载到: {local_dir}")