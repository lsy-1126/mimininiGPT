import json
import re

def clean(input_file, train_out, val_out):
    all_dialogues=[]
    with open(input_file,'r',encoding='utf-8')as f_in:
        for line in f_in:
            try:
                data=json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            instruction=data.get("instruction","")
            response=data.get("response","")
            instruction=re.sub(r"\s+"," ",instruction).strip()
            response=re.sub(r"\s+"," ",response).strip()
            if len(instruction) < 2 or len(response) < 2:
                continue
            format_text=f"<|User|>: {instruction}\n<|Bot|>: {response}\n<|End|>\n"
            all_dialogues.append(format_text)
    split_idx=int(len(all_dialogues)*0.9)
    train_data=all_dialogues[:split_idx]
    val_data=all_dialogues[split_idx:]
    with open(train_out,'w',encoding='utf-8') as f_out:
        f_out.write("".join(train_data))
    with open(val_out,'w',encoding='utf-8') as f_out:
        f_out.write("".join(val_data))
clean("databricks-dolly-15k.jsonl","train_chat.txt","val_chat.txt")




