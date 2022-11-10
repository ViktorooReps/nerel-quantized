import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch.cuda
from transformers import HfArgumentParser

from quant.model import SpanNERModel


@dataclass
class OptimizationArguments:
    model: Path = field(metadata={'help': 'Model .pkl file to optimize'})
    device: str = field(default='cpu', metadata={'help': 'Device to use for optimization. Options: cuda, cpu.'})

    prune: float = field(default=0.0, metadata={'help': 'Fraction of all heads to prune.'})
    dataset_dir: Path = field(default=Path('data'), metadata={'help': 'Dataset to use for pruning.'})

    onnx_dir: Path = field(default=Path('onnx'), metadata={'help': 'Path to directory where to store ONNX models.'})
    fuse: bool = field(default=False, metadata={'help': 'Fuse some elements of the model (is not supported with quantization)'})
    quantize: bool = field(default=False, metadata={'help': 'Quantize the model'})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = HfArgumentParser(dataclass_types=[OptimizationArguments])
    opt_args, = parser.parse_args_into_dataclasses()
    opt_args: OptimizationArguments

    model = SpanNERModel.load(opt_args.model)
    model.to(torch.device(opt_args.device))

    if opt_args.prune > 0:
        model.prune(opt_args.dataset_dir, opt_args.prune, batch_size=1)
        model.cpu()
        pruned_path = Path(opt_args.model.parent.joinpath('pruned_' + opt_args.model.name))
        model.save(pruned_path)
        logging.info(f'Model pruned and saved to {pruned_path}')

    model.optimize(opt_args.onnx_dir, fuse=opt_args.fuse, quant=opt_args.quantize)
    opt_path = opt_args.onnx_dir.joinpath('main.pkl')
    model.save(opt_path)
    logging.info(f'Model optimized and saved to {opt_path}')
