# Evaluation of BERT quantization for Named Entity Recognition

## Experiment results

| Model          | fused | quantized | pruned | Score (F1) | Time (s) |
|----------------|-------|-----------|--------|------------|----------|
| PyTorch RuBERT | No    | No        | 0%     | 67.26%     | 316      |
| ONNX RuBERT    | No    | Yes       | 0%     | 67.25%     | 368      |
| ONNX RuBERT    | No    | Yes       | 50%    | 59.17%     | 332      |
