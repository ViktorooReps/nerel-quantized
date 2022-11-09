# Evaluation of BERT quantization for Named Entity Recognition

## Experiment results

| Model          | fused | quantized | Score (F1) | Time (s) |
|----------------|-------|-----------|------------|----------|
| PyTorch RuBERT | No    | No        | 67.26%     | 316      |
| ONNX RuBERT    | No    | Yes       | 67.25%     | 368      |
