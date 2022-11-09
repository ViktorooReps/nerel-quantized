import logging
from dataclasses import dataclass
from enum import Enum
from functools import partial
from itertools import chain, starmap
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, Iterable, List, Set, Dict, Callable, Union

import torch
from torch import LongTensor, BoolTensor
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import EncodingFast

from quant.utils import pad_images, invert

logger = logging.getLogger(__name__)


class TypedSpan(NamedTuple):
    start: int
    end: int
    type: str


@dataclass
class Example:
    text_id: int
    example_start: int
    input_ids: LongTensor  # shape: (LENGTH)
    start_offset: LongTensor
    end_offset: LongTensor
    target_label_ids: Optional[LongTensor]  # shape: (LENGTH, LENGTH)


@dataclass
class BatchedExamples:
    text_ids: Tuple[int, ...]
    example_starts: Tuple[int, ...]
    input_ids: LongTensor  # shape: (BATCH_SIZE, LENGTH)
    start_offset: LongTensor
    end_offset: LongTensor
    padding_mask: BoolTensor  # shape: (BATCH_SIZE, LENGTH)


def collate_examples(
        examples: Iterable[Example],
        *,
        padding_id: int = -100,
        pad_length: Optional[int] = None
) -> Dict[str, Union[BatchedExamples, Optional[LongTensor]]]:

    all_text_ids: List[int] = []
    all_example_starts: List[int] = []
    all_input_ids: List[LongTensor] = []
    all_padding_masks: List[BoolTensor] = []
    all_start_offsets: List[LongTensor] = []
    all_end_offsets: List[LongTensor] = []
    target_label_ids: Optional[List[LongTensor]] = None

    no_target_label_ids: Optional[bool] = None

    for example in examples:
        all_text_ids.append(example.text_id)
        all_example_starts.append(example.example_start)
        all_input_ids.append(example.input_ids)
        all_start_offsets.append(example.start_offset)
        all_end_offsets.append(example.end_offset)
        all_padding_masks.append(torch.ones_like(example.input_ids, dtype=torch.bool).bool())

        if no_target_label_ids is None:
            no_target_label_ids = (example.target_label_ids is None)
            if not no_target_label_ids:
                target_label_ids: List[LongTensor] = []

        if (example.target_label_ids is None) != no_target_label_ids:
            raise RuntimeError('Inconsistent examples at collate_examples!')

        if example.target_label_ids is not None:
            target_label_ids.append(example.target_label_ids)

    return {
        'examples': BatchedExamples(
            tuple(all_text_ids),
            tuple(all_example_starts),
            pad_sequence(all_input_ids, batch_first=True, padding_value=padding_id).long(),
            pad_sequence(all_start_offsets, batch_first=True, padding_value=-100).long(),
            pad_sequence(all_end_offsets, batch_first=True, padding_value=-100).long(),
            pad_sequence(all_padding_masks, batch_first=True, padding_value=False).bool()
        ),
        'labels': pad_images(target_label_ids, padding_value=-100, padding_length=pad_length) if not no_target_label_ids else None
    }


def batch_examples(examples: Iterable[Example], *, batch_size: int = 1) -> Iterable[Dict[str, Union[BatchedExamples, LongTensor]]]:
    """Groups examples into batches."""
    curr_batch = []
    for example in examples:
        if len(curr_batch) == batch_size:
            yield collate_examples(curr_batch)
            curr_batch = []

        curr_batch.append(example)

    if len(curr_batch):
        yield collate_examples(curr_batch)


def convert_to_examples(
        text_id: int,
        encoding: EncodingFast,
        category_mapping: Dict[str, int],
        no_entity_category: str,
        *,
        max_length: int = 512,
        entities: Optional[Set[TypedSpan]] = None,
) -> Iterable[Example]:
    """Encodes entities and splits encoded text into chunks."""

    sequence_length = len(encoding.ids)
    offset = torch.tensor(encoding.offsets, dtype=torch.long)

    target_label_ids: Optional[LongTensor] = None
    if entities is not None:
        token_start_mapping = {}
        token_end_mapping = {}
        for token_idx, (token_start, token_end) in enumerate(encoding.offsets):
            token_start_mapping[token_start] = token_idx
            token_end_mapping[token_end] = token_idx

        no_entity_category_id = category_mapping[no_entity_category]
        category_id_mapping = invert(category_mapping)

        text_length = len(encoding.ids)
        target_label_ids = torch.full((text_length, text_length), fill_value=no_entity_category_id, dtype=torch.long).long()
        for start, end, category in entities:
            try:
                token_start = token_start_mapping[start]
            except KeyError:
                logger.warning(f'changing {start} to {start + 1}')
                token_start = token_end_mapping[start + 1]

            try:
                token_end = token_end_mapping[end]
            except KeyError:  # for some reason some ends are shifted by one
                logger.warning(f'changing {end} to {end + 1}')
                token_end = token_end_mapping[end + 1]

            if target_label_ids[token_start][token_end] != no_entity_category_id:
                from_category = category_id_mapping[target_label_ids[token_start][token_end].item()]
                logger.warning(f'Rewriting entity of category {from_category} with {category} at ({start} {end}) span')

            target_label_ids[token_start][token_end] = category_mapping[category]

    # split encoding into max_length-token chunks

    chunk_start = 0
    while chunk_start < sequence_length:
        chunk_end = min(chunk_start + max_length, sequence_length)
        ex = Example(
            text_id,
            chunk_start,
            torch.tensor(encoding.ids[chunk_start:chunk_end], dtype=torch.long).long(),
            offset[chunk_start:chunk_end, 0],
            offset[chunk_start:chunk_end, 1],
            target_label_ids[chunk_start:chunk_end, chunk_start:chunk_end] if target_label_ids is not None else None
        )
        yield ex
        chunk_start = chunk_end


def convert_all_to_examples(
        encodings: Iterable[EncodingFast],
        category_mapping: Dict[str, int],
        no_entity_category: str,
        *,
        max_length: int = 512
) -> Iterable[Example]:

    converter = partial(
        convert_to_examples,
        category_mapping=category_mapping,
        no_entity_category=no_entity_category,
        max_length=max_length
    )
    # noinspection PyTypeChecker
    return chain.from_iterable(starmap(converter, enumerate(encodings)))


class DatasetType(str, Enum):
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'


def read_annotation(annotation_file: Path) -> Set[TypedSpan]:
    collected_annotations: Set[TypedSpan] = set()
    with open(annotation_file) as f:
        for line in f:
            if line.startswith('T'):
                _, span_info, value = line.strip().split('\t')

                if ';' not in span_info:  # skip multispan
                    category, start, end = span_info.split(' ')
                    collected_annotations.add(TypedSpan(int(start), int(end), category))

    return collected_annotations


def read_text(text_file: Path) -> str:
    with open(text_file) as f:
        return f.read()


def collect_categories(annotation_files: Iterable[Path]) -> Set[str]:
    all_annotations = list(map(read_annotation, annotation_files))

    all_categories: Set[str] = set()
    for document_annotations in all_annotations:
        all_categories.update(map(lambda span: span.type, document_annotations))

    return all_categories


def get_dataset_files(
        dataset_dir: Path,
        dataset_type: DatasetType,
        *,
        exclude_filenames: Set[str] = None
) -> Tuple[List[Path], List[Path]]:

    if exclude_filenames is None:
        exclude_filenames = set()

    dataset_dir = dataset_dir.joinpath(dataset_type.value)

    if not dataset_dir.exists():
        raise RuntimeError(f'Dataset directory {dataset_dir} does not exist!')

    if not dataset_dir.is_dir():
        raise RuntimeError(f'Provided path {dataset_dir} is not a directory!')

    def is_not_excluded(file: Path) -> bool:
        return file.with_suffix('').name not in exclude_filenames

    return sorted(filter(is_not_excluded, dataset_dir.glob('*.txt'))), sorted(filter(is_not_excluded, dataset_dir.glob('*.ann')))


def read_nerel(
        dataset_dir: Path,
        dataset_type: DatasetType,
        tokenizer: Callable[[List[str]], List[EncodingFast]],
        category_mapping: Dict[str, int],
        no_entity_category: str,
        *,
        exclude_filenames: Set[str] = None
) -> Iterable[Example]:

    text_files, annotation_files = get_dataset_files(dataset_dir, dataset_type, exclude_filenames=exclude_filenames)

    all_annotations = list(map(read_annotation, annotation_files))
    all_texts = list(map(read_text, text_files))

    encodings = tokenizer(all_texts)

    for text_id, (encoding, entities) in enumerate(zip(encodings, all_annotations)):
        yield from convert_to_examples(text_id, encoding, category_mapping, entities=entities, no_entity_category=no_entity_category)
