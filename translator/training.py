import math
import os
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm
import translator.custom_dataloader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import get_scheduler
from transformers import AdamW
from torch.utils.data import DataLoader

from translator.helpers import load_file_content, get_dataset_info
from translator.model import TransformerModel

TOKENIZER_CHECKPOINT = "t5-small"
BATCH_SIZE = 8
MAX_SEQ_LEN = 512
INPUT_VOCAB_SIZE = 512
OUTPUT_VOCAB_SIZE = 512
ENCODER_LAYER = 6
DECODER_LAYER = 6
ENCODER_HEADS = 2
DECODER_HEADS = 2
EMBEDDING_SIZE = 512
FEED_FORWARD_SIZE = 1024
DROPOUT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_custom_tokenizer(data_folder: str, checkpoint: str) -> Any:
    dataset_info = get_dataset_info(data_folder)
    vocab_size = dataset_info["vocabulary_size"]
    old_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return old_tokenizer.train_new_from_iterator(load_file_content(data_folder), int(vocab_size))


# data_folder = "/media/md0/home/armel/transformers/translator/translator/resources"
data_folder = "/home/armel/Code/reciTAL/Proto/Doc-ai/translator/translator/resources"
tokenizer = build_custom_tokenizer(data_folder, checkpoint=TOKENIZER_CHECKPOINT)

dataset_info = get_dataset_info(data_folder)
vocab_size = dataset_info["vocabulary_size"]
in_max_seq = dataset_info["in_max_seq"]
out_max_seq = dataset_info["out_max_seq"]
max_seq = max(in_max_seq, out_max_seq)


def tokenizer_samples(examples):
    src_input = tokenizer(examples["src"], max_length=max_seq, padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        target_input = tokenizer(examples["trg"], max_length=max_seq, padding="max_length", truncation=True)

    src_input["trg_input_ids"] = target_input["input_ids"]
    src_input["trg_mask"] = target_input["attention_mask"]

    return src_input


loader_path = os.path.abspath(translator.custom_dataloader.__file__)
raw_datasets = load_dataset(
        path=loader_path,
        name="translator_dataset_builder",
        data_folder=data_folder,
    )

tokenized_dataset = raw_datasets.map(
    tokenizer_samples,
    batched=True,
    remove_columns=["id", "src", "trg"]
)

transformer_model = TransformerModel(
    encoder_layer=ENCODER_LAYER,
    decoder_layer=DECODER_LAYER,
    encoder_heads=ENCODER_HEADS,
    decoder_heads=DECODER_HEADS,
    input_vocab_size=vocab_size,
    embeddings_len=EMBEDDING_SIZE,
    feedforward_len=FEED_FORWARD_SIZE,
    target_vocab_size=vocab_size,
    device=DEVICE,
    dropout=DROPOUT
)
transformer_model.to(DEVICE)


# Initialize Params
for param in transformer_model.parameters():
    if param.dim() > 1:
        nn.init.xavier_uniform_(param)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(
    tokenized_dataset["train"], shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_dataset["validation"], batch_size=BATCH_SIZE, collate_fn=data_collator
)

optimizer = AdamW(transformer_model.parameters(), lr=5e-5)
loss_function = nn.CrossEntropyLoss(ignore_index = 0) # tokenizer.pad_token)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))


def train_epoch(transformer_model, train_dataloader, optimizer, loss_function, clip):
    transformer_model.train()
    epoch_loss = 0

    for batch_index, batch in enumerate(train_dataloader):
        print(f"Batch N° {batch_index+1}")
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        outputs = transformer_model(**batch)

        trg_size_1 = batch["trg_input_ids"].size()[0]
        trg_size_2 = batch["trg_input_ids"].size()[1]

        loss = loss_function(outputs.view(-1, vocab_size), batch["trg_input_ids"].view(trg_size_1*trg_size_2))
        print(f"Loss : {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), clip)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    return epoch_loss/len(train_dataloader)


def evaluate_epoch(transfomer_model, eval_dataloader, loss_function):
    transfomer_model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            outputs = transformer_model(**batch)

            loss = torch.tensor(0.0, requires_grad=True, device=DEVICE)
            for pred, truth in zip(outputs, batch["trg_input_ids"]):
                loss = loss_function(pred, truth)

            epoch_loss += loss.item()

    return epoch_loss/len(eval_dataloader)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(transformer_model):,} trainable parameters')

best_valid_loss = float('inf')

model_file = "fra-nufi-model.pt"

for epoch in range(num_epochs):
    print(f"Epoch : {epoch}")
    train_loss = train_epoch(transformer_model, train_dataloader, optimizer, loss_function, clip=1)
    eval_loss = evaluate_epoch(transformer_model, eval_dataloader, loss_function)

    if eval_loss < best_valid_loss:
        best_valid_loss = eval_loss
        torch.save(transformer_model.state_dict(), model_file)

    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {eval_loss:.3f} |  Val. PPL: {math.exp(eval_loss):7.3f}')

