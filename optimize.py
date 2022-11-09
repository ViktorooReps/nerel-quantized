from pathlib import Path

import torch.cuda

from quant.model import SpanNERModel

if __name__ == '__main__':
    model = SpanNERModel.load(Path('model_10epochs_cpu.pkl'))
    model.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.prune(Path('data'), 0.2)
    model.cpu()
    model.save(Path('model_10epochs_cpu_pruned.pkl'))
    print('Model pruned and saved!')
    model.optimize(Path('onnx'), fuse=False, quant=True)
    model.save(Path('onnx/main.pkl'))
