import os
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset

from translator.helpers import load_file_content, get_dataset_info
import translator.custom_dataloader
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

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

test_dataloader = DataLoader(
    tokenized_dataset["test"], batch_size=8, collate_fn=data_collator
)


def translate_sentences(transformer_model, test_dataloader, tokenizer):
    transformer_model.eval()

    for batch in test_dataloader:
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        with torch.no_grad():
            outputs = transformer_model(**batch)
            print(f"out dim : {outputs.size()}")
            out_word_ids = torch.argmax(outputs, -1)

            print(f"Word IDs : {out_word_ids}")
            for out_seq, input_ids, out_ids in zip(out_word_ids, batch["input_ids"],  batch["trg_input_ids"]):
                out_text = tokenizer.decode(out_seq, skip_special_tokens=True)
                in_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                target_seq = tokenizer.decode(out_ids, skip_special_tokens=True)
                print()
                print("--------------------------")
                print(f"Input : {in_text}")
                print(f"Target : {target_seq}")
                print(f"Predict : {out_text}")
                print("------------------------")
                print()


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

model_file = "fra-nufi-model.pt"
transformer_model.load_state_dict(torch.load(model_file))


translate_sentences(transformer_model, test_dataloader, tokenizer)
