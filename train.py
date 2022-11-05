from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from torch.utils.data import Dataset
from transformers import Trainer, HfArgumentParser, TrainingArguments
from transformers.modeling_utils import unwrap_model

from quant.datamodel import read_nerel, collect_category_mapping, get_label_mapping, DatasetType, Example, collate_examples
from quant.model import NERModel, ModelArguments


@dataclass
class DatasetArguments:
    dataset_dir: Path = field(metadata={'help': 'NEREL dataset directory with train/dev/test subdirectories.'})


class NERDataset(Dataset[Example]):

    def __init__(self, examples: Iterable[Example]):
        self._examples = list(examples)

    def __getitem__(self, index) -> Example:
        return self._examples[index]

    def __len__(self):
        return len(self._examples)


if __name__ == '__main__':
    parser = HfArgumentParser(dataclass_types=[ModelArguments, DatasetArguments, TrainingArguments])
    model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    training_args: TrainingArguments
    dataset_args: DatasetArguments
    model_args: ModelArguments

    category_mapping = collect_category_mapping(dataset_args.dataset_dir)
    model = NERModel(model_args, category_mapping, get_label_mapping())

    train_dataset = NERDataset(read_nerel(dataset_args.dataset_dir, DatasetType.TRAIN, model.tokenize, category_mapping))
    dev_dataset = NERDataset(read_nerel(dataset_args.dataset_dir, DatasetType.DEV, model.tokenize, category_mapping))

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_examples,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=None,  # TODO
        preprocess_logits_for_metrics=None,  # TODO
        callbacks=None  # TODO
    )
    trainer.train()

    # noinspection PyTypeChecker
    trained_model: NERModel = unwrap_model(trainer.model_wrapped)
    trained_model.save(model_args.save_path)
