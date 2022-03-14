import os
from typing import List


def load_file_content(data_folder: str, pair_seperator="\t") -> List[str]:
    processed_files = [
        os.path.join(data_folder, "train.txt"),
        os.path.join(data_folder, "validation.txt"),
        os.path.join(data_folder, "test.txt")
    ]
    for file_path in processed_files:
        with open(file_path, encoding="utf-8") as file_:
            lines = file_.readlines()

        for line in lines:
            sent = line.split(pair_seperator)
            for item in sent:
                yield item.replace("\n", "").strip()


def get_dataset_info(data_folder: str, pair_seperator="\t") -> dict:
    processed_files = [
        os.path.join(data_folder, "train.txt"),
        os.path.join(data_folder, "validation.txt"),
        os.path.join(data_folder, "test.txt")
    ]
    tokens = []
    in_seq_len = -10
    out_seq_len = -10
    for file_path in processed_files:
        with open(file_path, encoding="utf-8") as file_:
            lines = file_.readlines()

        for line in lines:
            sent = line.split(pair_seperator)
            if len(sent) > 1:
                if len(sent[0]) > in_seq_len:
                    in_seq_len = len(sent[0])

                if len(sent[1]) > out_seq_len:
                    out_seq_len = len(sent[1])

            for item in sent:
                tokens.append(item.replace("\n", "").strip().lower())

    return {
        "vocabulary_size": len(list(set(tokens))),
        "in_max_seq": in_seq_len + 5,
        "out_max_seq": out_seq_len + 5
    }


