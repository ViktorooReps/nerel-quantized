import logging
from dataclasses import dataclass, field
from pathlib import Path

from transformers import HfArgumentParser

from quant.datamodel import DatasetType, get_dataset_files, read_text, read_annotation

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerArguments:
    category: str = field(default=None, metadata={'help': 'Category to analyze.'})
    dataset_dir: Path = field(default=Path('data'), metadata={'help': 'Dataset directory.'})
    dataset_type: str = field(default='train', metadata={'help': 'Dataset type to analyze.'})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = HfArgumentParser(dataclass_types=[AnalyzerArguments])
    an_args, = parser.parse_args_into_dataclasses()
    an_args: AnalyzerArguments

    if an_args.category is None:
        logger.info('Nothing to analyze')
    else:
        text_files, annotation_files = get_dataset_files(an_args.dataset_dir, DatasetType(an_args.dataset_type))

        for text, annotation, file in zip(map(read_text, text_files), map(read_annotation, annotation_files), text_files):
            for start, end, category in annotation:
                if category.lower() == an_args.category.lower():
                    logger.info(f'File: {file.name}')
                    logger.info(f'{text[:start]}__{text[start:end]}__{text[end:]}')
                    input()
