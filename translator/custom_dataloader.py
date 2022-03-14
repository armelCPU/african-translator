import logging
import os

import datasets

_CITATION = "translation fr-fon dataset"
_DESCRIPTION = """\
Dataset for translation task
"""


_SPLIT_FILES = {
    "train": f"train.txt",
    "validation": f"validation.txt",
    "test": f"test.txt"
}


def read_dataset_file(parallel_file, pair_seperator="\t", reverse=False):
    with open(parallel_file, encoding="utf-8") as file_:
        lines = file_.readlines()

    pairs = []
    for line in lines:
        sentence = line.split(pair_seperator)
        if len(sentence) < 2:
            continue
        else:
            right_sent = sentence[1].replace("\n", "")
            left_sent = sentence[0].replace("\n", "")

        if not reverse:
            pairs.append([left_sent, right_sent])
        else:
            pairs.append([right_sent, left_sent])

    logging.info(f"Number of sentences in dataset : {len(pairs)}")
    return pairs


class TranslationDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for translation dataset"""

    def __init__(self, data_folder: str="", **kwargs):
        """BuilderConfig for The translation dataset.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TranslationDatasetConfig, self).__init__(**kwargs)
        self.data_folder = data_folder


class TranslationDatasetBuilder(datasets.GeneratorBasedBuilder):
    """in-house dataset."""

    BUILDER_CONFIGS = [
        TranslationDatasetConfig(
            name="translator_dataset_builder",
            version=datasets.Version("1.0.0"),
            description=_DESCRIPTION
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "src": datasets.Value("string"),
                    "trg": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_DESCRIPTION,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(self.config.data_folder, _SPLIT_FILES["train"])},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(self.config.data_folder, _SPLIT_FILES["validation"])},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(self.config.data_folder, _SPLIT_FILES["test"])},
            ),
        ]

    def _generate_examples(self, filepath):
        logging.info("⏳ Generating examples from = %s", filepath)

        pairs = read_dataset_file(filepath)
        for id_, pair in enumerate(pairs[:7]):
            yield id_, {
                "id": str(id_),
                "src": pair[0],
                "trg": pair[1]
            }
