from pathlib import Path

from quant.model import SpanNERModel

if __name__ == '__main__':
    model = SpanNERModel.load(Path('model_10epochs_cpu.pkl'))
    model.prune(Path('data'), 0.2)
    model.save(Path('model_10epochs_cpu_pruned.pkl'))
    model.optimize(Path('onnx'), fuse=False, quant=True)
    model.save(Path('onnx/main.pkl'))
