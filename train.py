from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Iterable, Dict

from sklearn.metrics import f1_score, recall_score, precision_score
from torch import Tensor, LongTensor
from torch.nn.functional import pad
from torch.utils.data import Dataset
from transformers import Trainer, HfArgumentParser, TrainingArguments, EvalPrediction
from transformers.modeling_utils import unwrap_model

from quant.datamodel import read_nerel, DatasetType, Example, collate_examples, collect_categories
from quant.model import ModelArguments, SpanNERModel


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


def compute_metrics(
        evaluation_results: EvalPrediction,
        category_id_mapping: Dict[int, str],
        no_entity_category_id: int
) -> Dict[str, float]:

    mask = (evaluation_results.label_ids != -100)

    label_ids = evaluation_results.label_ids[mask]
    predictions = evaluation_results.predictions[mask]

    labels = sorted(category_id_mapping.keys())
    f1_category_scores = f1_score(label_ids, predictions, average=None, labels=labels)
    recall_category_scores = recall_score(label_ids, predictions, average=None, labels=labels)
    precision_category_scores = precision_score(label_ids, predictions, average=None, labels=labels)

    results: Dict[str, float] = {}
    sum_f1 = 0
    sum_recall = 0
    sum_precision = 0
    for category_id, (f1, recall, precision) in enumerate(zip(f1_category_scores, recall_category_scores, precision_category_scores)):
        if category_id == no_entity_category_id:
            continue

        category = category_id_mapping[category_id]
        results[f'F1_{category}'] = f1
        results[f'Recall_{category}'] = recall
        results[f'Precision_{category}'] = precision

        sum_f1 += f1
        sum_recall += recall
        sum_precision += precision

    num_categories = len(category_id_mapping) - 1

    results['F1_macro'] = sum_f1 / num_categories
    results['Recall_macro'] = sum_recall / num_categories
    results['Precision_macro'] = sum_precision / num_categories
    return results


def pad_predictions(predictions: Tensor, _: LongTensor, *, padding_length: int) -> Tensor:
    _, curr_length, _ = predictions.shape
    return pad(predictions, [  # IDK why tf it works
        0, padding_length - curr_length,
        0, padding_length - curr_length
    ], value=-100)


if __name__ == '__main__':
    parser = HfArgumentParser(dataclass_types=[ModelArguments, DatasetArguments, TrainingArguments])
    model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    training_args: TrainingArguments
    dataset_args: DatasetArguments
    model_args: ModelArguments

    categories = collect_categories(dataset_args.dataset_dir)
    model = SpanNERModel(model_args, categories)

    train_dataset = NERDataset(list(read_nerel(
        dataset_args.dataset_dir, DatasetType.TRAIN, model.tokenize, model.category_mapping, model.no_entity_category
    ))[:10])
    dev_dataset = NERDataset(list(read_nerel(
        dataset_args.dataset_dir, DatasetType.DEV, model.tokenize, model.category_mapping, model.no_entity_category
    ))[:10])

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=partial(collate_examples, padding_id=model.pad_token_id, pad_length=model.context_length),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=partial(
            compute_metrics,
            category_id_mapping=model.category_id_mapping,
            no_entity_category_id=model.no_entity_category_id
        ),
        preprocess_logits_for_metrics=partial(pad_predictions, padding_length=model.context_length),
        callbacks=None  # TODO
    )
    trainer.train()

    # noinspection PyTypeChecker
    trained_model: SpanNERModel = unwrap_model(trainer.model_wrapped)
    trained_model.save(model_args.save_path)
