# Evaluation of BERT quantization for Named Entity Recognition

## Experiment results

| Model                           | fused | quantized | pruned | Score (F1) | Time (s) |
|---------------------------------|-------|-----------|--------|------------|----------|
| PyTorch RuBERT                  | No    | No        | 0%     | 67.26%     | 316      |
| ONNX RuBERT                     | No    | Yes       | 0%     | 67.25%     | 368      |
| ONNX RuBERT                     | No    | Yes       | 50%    | 59.17%     | 332      |
| PyTorch Truncated LaBSE (en-ru) | No    | No        | 0%     | 70.01%     | 317      |
| ONNX Truncated LaBSE (en-ru)    | No    | Yes       | 0%     | 13.91%     | 374      |
