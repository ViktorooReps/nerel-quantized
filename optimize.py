from pathlib import Path

from quant.model import SpanNERModel

if __name__ == '__main__':
    model = SpanNERModel.load(Path('model_cpu.pkl'))
    model.optimize(Path('onnx'), fuse=True, quant=True)
