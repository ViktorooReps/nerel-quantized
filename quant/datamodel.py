import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import partial
from itertools import chain, starmap
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, Iterable, List, Set, Dict, Callable, DefaultDict

import torch
from torch import LongTensor, BoolTensor
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import EncodingFast


logger = logging.getLogger(__name__)


class TypedSpan(NamedTuple):
    start: int
    end: int
    type: str


@dataclass
class Example:
    text_id: int
    example_start: int
    input_ids: LongTensor  # shape: (SEQUENCE_LENGTH)
    target_label_ids: Optional[LongTensor]  # shape: (NUM_CATEGORIES, SEQUENCE_LENGTH)


@dataclass
class BatchedExamples:
    text_ids: Tuple[int, ...]
    example_starts: Tuple[int, ...]
    input_ids: LongTensor  # shape: (BATCH_SIZE, SEQUENCE_LENGTH)
    padding_mask: BoolTensor  # shape: (BATCH_SIZE, SEQUENCE_LENGTH)
    target_label_ids: Optional[LongTensor]  # shape: (BATCH_SIZE, NUM_CATEGORIES, SEQUENCE_LENGTH)


def collate_examples(examples: Iterable[Example]) -> BatchedExamples:
    all_text_ids: List[int] = []
    all_example_starts: List[int] = []
    all_input_ids: List[LongTensor] = []
    all_padding_masks: List[BoolTensor] = []
    target_label_ids: Optional[List[LongTensor]] = None

    no_target_label_ids: Optional[bool] = None

    for example in examples:
        all_text_ids.append(example.text_id)
        all_example_starts.append(example.example_start)
        all_input_ids.append(example.input_ids)
        all_padding_masks.append(torch.ones_like(example.input_ids, dtype=torch.bool).bool())

        if no_target_label_ids is None:
            no_target_label_ids = (example.target_label_ids is None)
            if not no_target_label_ids:
                target_label_ids: List[LongTensor] = []

        if (example.target_label_ids is None) != no_target_label_ids:
            raise RuntimeError('Inconsistent examples at collate_examples!')

        if example.target_label_ids is not None:
            target_label_ids.append(example.target_label_ids)

    return BatchedExamples(
        tuple(all_text_ids),
        tuple(all_example_starts),
        pad_sequence(all_input_ids, batch_first=True, padding_value=-100).long(),
        pad_sequence(all_padding_masks, batch_first=True, padding_value=False).bool(),
        pad_sequence(target_label_ids, batch_first=True, padding_value=-100) if not no_target_label_ids else None
    )


def batch_examples(examples: Iterable[Example], *, batch_size: int = 1) -> Iterable[BatchedExamples]:
    """Groups examples into batches."""
    curr_batch = []
    for example in examples:
        if len(curr_batch) == batch_size:
            yield collate_examples(curr_batch)
            curr_batch = []

        curr_batch.append(example)

    if len(curr_batch):
        yield collate_examples(examples)


def convert_to_examples(
        text_id: int,
        encoding: EncodingFast,
        category_mapping: Dict[str, int],
        label_mapping: Dict[str, int],
        *,
        max_length: int = 512,
        entities: Optional[Set[TypedSpan]] = None,
) -> Iterable[Example]:
    """Encodes entities and splits encoded text into chunks."""

    sequence_length = len(encoding.ids)

    target_label_ids: Optional[LongTensor] = None
    if entities is not None:
        # group entities into categories
        category_spans: DefaultDict[str, Set[Tuple[int, int]]] = defaultdict(set)
        for start, end, type_ in entities:
            category_spans[type_].add((start, end))

        # by default set every label to O
        num_categories = len(category_mapping)
        target_label_ids: LongTensor = torch.full((num_categories, sequence_length), fill_value=label_mapping['O'], dtype=torch.long).long()

        # collect mappings (position in text) -> (position in label_ids)
        token_start_mapping: Dict[int, int] = {}
        token_end_mapping: Dict[int, int] = {}
        for token_position, (orig_start, orig_end) in enumerate(encoding.offsets):
            token_start_mapping[orig_start] = token_position
            token_end_mapping[orig_end] = token_position + 1

        def filter_non_intersecting(spans_: Iterable[Tuple[int, int]]) -> Iterable[Tuple[int, int]]:
            """Filter non-intersecting spans and sort by span start."""
            spans_ = sorted(spans_)
            prev_end = 0
            for s_start, s_end in spans_:
                if s_start < prev_end:
                    logger.warning(f'Removed intersecting spans for {category} category!')
                    continue
                yield s_start, s_end
                prev_end = s_end

        def is_entity_label_id(label_id: int):
            return label_id == label_mapping['I'] or label_id == label_mapping['B']

        for category, spans in category_spans.items():
            category_id = category_mapping[category]
            category_label_ids = target_label_ids[category_id]

            for span_start, span_end in filter_non_intersecting(spans):
                span_start_position = token_start_mapping[span_start]
                span_end_position = token_end_mapping[span_start]

                category_label_ids[span_start_position:span_end_position] = label_mapping['I']

                # add B to separate (following the IOB scheme)
                if span_start_position and is_entity_label_id(category_label_ids[span_start_position - 1]):
                    category_label_ids[span_start_position] = label_mapping['B']

    # split encoding into max_length-token chunks

    chunk_start = 0
    while chunk_start < sequence_length:
        chunk_end = min(chunk_start + max_length, sequence_length)
        yield Example(
            text_id,
            chunk_start,
            encoding.ids[chunk_start:chunk_end],
            target_label_ids[chunk_start:chunk_end] if target_label_ids is not None else None
        )
        chunk_start = chunk_end


def convert_all_to_examples(
        encodings: Iterable[EncodingFast],
        category_mapping: Dict[str, int],
        label_mapping: Dict[str, int],
        *,
        max_length: int = 512
) -> Iterable[Example]:

    converter = partial(convert_to_examples, category_mapping=category_mapping, label_mapping=label_mapping, max_length=max_length)
    # noinspection PyTypeChecker
    return chain.from_iterable(starmap(converter, enumerate(encodings)))


def decode(encoding: EncodingFast, labels: Iterable[str], category: str) -> Set[TypedSpan]:
    """Decodes examples following IOB labelling strategy."""
    entity_start: Optional[int] = None
    entities: Set[TypedSpan] = set()

    token_end = None
    for label, (token_start, token_end) in zip(labels, encoding.offsets):
        if label == 'B':
            if entity_start is not None:
                entity_end = token_start
                entities.add(TypedSpan(entity_start, entity_end, category))
            entity_start = token_start
            continue

        if label == 'I':
            if entity_start is None:
                entity_start = token_start
            continue

        # label == 'O'
        if entity_start is not None:
            entity_end = token_start
            entities.add(TypedSpan(entity_start, entity_end, category))
            entity_start = None

    if entity_start is not None:
        entity_end = token_end
        entities.add(TypedSpan(entity_start, entity_end, category))

    return entities


class DatasetType(str, Enum):
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'


def get_label_mapping() -> Dict[str, int]:
    return {'O': 0, 'I': 1, 'B': 2}  # IOB


def read_annotation(annotation_file: Path) -> Set[TypedSpan]:
    collected_annotations: Set[TypedSpan] = set()
    with open(annotation_file) as f:
        for line in f:
            type_, span_info, value = line.split('\t')

            type_: str
            if type_.startswith('T') and ';' not in span_info:  # skip multispan
                category, start, end = span_info.split(' ')
                collected_annotations.add(TypedSpan(int(start), int(end), category))

    return collected_annotations


def collect_category_mapping(dataset_dir: Path) -> Dict[str, int]:
    dataset_dir = dataset_dir.joinpath(DatasetType.TRAIN)

    if not dataset_dir.exists():
        raise RuntimeError(f'Dataset directory {dataset_dir} does not exist!')

    if not dataset_dir.is_dir():
        raise RuntimeError(f'Provided path {dataset_dir} is not a directory!')

    annotation_files = dataset_dir.glob('*.ann')
    all_annotations = list(map(read_annotation, annotation_files))

    all_categories: Set[str] = set()
    for document_annotations in all_annotations:
        all_categories.update(map(lambda span: span.type, document_annotations))

    # noinspection PyTypeChecker
    return dict(enumerate(all_categories))


def read_nerel(
        dataset_dir: Path,
        dataset_type: DatasetType,
        tokenizer: Callable[[List[str]], List[EncodingFast]],
        category_mapping: Dict[str, int],
        *,
        exclude_filenames: Set[str] = None
) -> Iterable[Example]:

    if exclude_filenames is None:
        exclude_filenames = set()

    dataset_dir = dataset_dir.joinpath(dataset_type)

    if not dataset_dir.exists():
        raise RuntimeError(f'Dataset directory {dataset_dir} does not exist!')

    if not dataset_dir.is_dir():
        raise RuntimeError(f'Provided path {dataset_dir} is not a directory!')

    def is_not_excluded(file: Path) -> bool:
        return file.with_suffix('').name not in exclude_filenames

    annotation_files = sorted(filter(is_not_excluded, dataset_dir.glob('*.ann')))
    text_files = sorted(filter(is_not_excluded, dataset_dir.glob('*.txt')))

    def read_text(text_file: Path) -> str:
        with open(text_file) as f:
            return f.read()

    all_annotations = list(map(read_annotation, annotation_files))
    all_texts = list(map(read_text, text_files))

    encodings = tokenizer(all_texts)

    for text_id, (encoding, entities) in enumerate(zip(encodings, all_annotations)):
        yield from convert_to_examples(text_id, encoding, category_mapping, get_label_mapping(), entities=entities)
