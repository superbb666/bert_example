from transformers import TrainingArguments, BertForSequenceClassification, BertTokenizerFast, Trainer, pipeline
import math 
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional

# args
training_args = TrainingArguments(
    output_dir='temp_trainer',
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    evaluation_strategy='epoch',
    num_train_epochs=5,
    load_best_model_at_end=True,
)
model_dir = 'temp_trainer'
# tokenizer and model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizerFast.from_pretrained(model_dir)  #, max_len=512, return_tensors='pt'
model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=2)
model.to(device)

# dataset
class BertPairDataset(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int):
        # 文件形式：0\tI am\tHe is a boy\n    1\tHe like it\tI am so happy\n
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        raw_text = [line.split('\t')[1:] for line in lines]
        label = [int(line.split('\t')[0]) for line in lines]
        batch_encoding = tokenizer(raw_text, add_special_tokens=True, truncation=True, max_length=block_size)  # , padding=True, return_tensors='pt'
        self.examples = []
        for i in range(len(label)):
            input_ids = torch.tensor(batch_encoding['input_ids'][i], dtype=torch.long)
            token_type_ids = torch.tensor(batch_encoding['token_type_ids'][i], dtype=torch.long)
            attention_mask = torch.tensor(batch_encoding['attention_mask'][i], dtype=torch.long)
            labels = torch.tensor(label[i], dtype=torch.long)
            self.examples.append({'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask, 'labels': labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

train_dataset = BertPairDataset(tokenizer, file_path='train_file.txt', block_size=256)
eval_dataset = BertPairDataset(tokenizer, file_path='train_file.txt', block_size=256)


# Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= train_dataset,
    eval_dataset= eval_dataset,
    tokenizer=tokenizer,
)
train_result = trainer.train()
trainer.save_model()  # Saves the tokenizer too for easy upload
print("train_result:", train_result)

# Evaluation
eval_output = trainer.evaluate()
perplexity = math.exp(eval_output["eval_loss"])
print("perplexity", perplexity)

# inference
fill_mask = pipeline(
    "sentiment-analysis",
    model= training_args.output_dir,
    tokenizer=tokenizer
)

inf_result = fill_mask(["He like it", "I am so happy"])
print('----inf_result---:', inf_result)
