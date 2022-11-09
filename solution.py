import time
from pathlib import Path
from typing import List, Iterable, Set, Tuple, Dict

import numpy as np

from quant.datamodel import TypedSpan, get_dataset_files, DatasetType, read_annotation, read_text
from quant.model import SpanNERModel


class Solution:

    model_path = Path('model.pkl')
    exclude_filenames = {
        '165459_text', '176167_text', '178485_text', '192238_text',
        '193267_text', '193946_text', '194112_text', '2021',
        '202294_text', '2031', '209438_text', '209731_text', '546860_text'
    }

    @classmethod
    def predict(cls, texts: List[str]) -> Iterable[Set[Tuple[TypedSpan]]]:
        model = SpanNERModel.load(cls.model_path)
        return model.predict(texts)

    @classmethod
    def evaluate(cls):
        text_files, annotation_files = get_dataset_files(Path('data'), DatasetType.TEST, exclude_filenames=cls.exclude_filenames)
        ground_truth = list(map(read_annotation, annotation_files))
        texts = list(map(read_text, text_files))

        model = SpanNERModel.load(cls.model_path)
        model_predictions = model.predict(texts)

        n_categories = len(model.category_mapping)

        def group_by_category(entities: Set[TypedSpan]) -> Dict[str, Set[Tuple[int, int]]]:
            groups = {cat: set() for cat in model.category_mapping.keys()}
            for entity in entities:
                groups[entity.type].add((entity.start, entity.end))
            return groups

        true_positives = np.zeros(n_categories, dtype=int)
        false_positives = np.zeros(n_categories, dtype=int)
        false_negatives = np.zeros(n_categories, dtype=int)

        for true_entities, predicted_entities in zip(ground_truth, model_predictions):
            true_groups = group_by_category(true_entities)
            predicted_groups = group_by_category(predicted_entities)

            for category in model.category_mapping.keys():
                category_id = model.category_mapping[category]
                true_set = true_groups[category]
                predicted_set = predicted_groups[category]

                true_positives[category_id] += len(true_set.union(predicted_set))
                false_positives[category_id] += len(predicted_set.difference(true_set))
                false_negatives[category_id] += len(true_set.difference(predicted_set))

        category_f1 = true_positives / (true_positives + (false_positives + false_negatives) / 2)
        for category_id, f1 in enumerate(category_f1):
            category = model.category_id_mapping[category_id]
            print(f'{category}: {f1 * 100:.2f}%')

        f1_macro = category_f1.mean()
        print(f'macro F1: {f1_macro * 100:.2f}%')

        return f1_macro


if __name__ == '__main__':
    start_time = time.time()
    Solution.evaluate()
    end_time = time.time()

    print(f'Test time: {end_time - start_time}s')
