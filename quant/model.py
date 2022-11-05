import logging
import pickle
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from os import environ, cpu_count
from pathlib import Path
from typing import TypeVar, Tuple, Iterable, List, Type, Optional, Set, Dict

import numpy as np
import torch
from torch import LongTensor, BoolTensor, Tensor, softmax
from torch.nn import Linear, Module, Dropout, Parameter, ModuleList
from torch.onnx import export
from transformers import AutoModel, PreTrainedModel, AutoTokenizer, PreTrainedTokenizer, TensorType
from transformers.convert_graph_to_onnx import quantize
from transformers.modeling_outputs import BaseModelOutput
from transformers.onnx import FeaturesManager, OnnxConfig
from transformers.tokenization_utils_base import EncodingFast

from quant.datamodel import TypedSpan, batch_examples, decode, convert_all_to_examples, BatchedExamples

# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
environ["OMP_NUM_THREADS"] = str(cpu_count())
environ["OMP_WAIT_POLICY"] = 'ACTIVE'

try:
    from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
    from onnxruntime_tools import optimizer
    from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions
except ImportError:
    raise ImportError('Could not import ONNX inference tools!')

_ModelType = TypeVar('_ModelType', bound=Module)

logger = logging.getLogger(__name__)


class SerializableModel(Module):

    def __init__(self):
        super().__init__()
        self._dummy_param = Parameter(torch.empty())

    @property
    def device(self) -> torch.device:
        return self._dummy_param.device

    def save(self, save_path: Path) -> None:
        previous_device = self.device
        self.cpu()
        with open(save_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.to(previous_device)

    @classmethod
    def load(cls: Type[_ModelType], load_path: Path) -> _ModelType:
        with open(load_path, 'rb') as f:
            return pickle.load(f)


class PredictionsCollector:

    def __init__(self, text_token_lengths: Iterable[int]):
        self._text_predictions: List[LongTensor] = list(map(partial(np.full, fill_value=-1, dtype=int), text_token_lengths))

    def add(self, text_id: int, start: int, predictions: LongTensor) -> None:
        self._text_predictions[text_id][start:start + len(predictions)] = predictions

    @property
    def predictions(self) -> List[LongTensor]:
        return self._text_predictions


class DotProductAttention(Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, query: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        return context, attn


_K = TypeVar('_K')
_V = TypeVar('_V')


def invert(d: Dict[_K, _V]) -> Dict[_V, _K]:
    return {v: k for k, v in d.items()}


@dataclass
class ModelArguments:
    bert_model: str = field(metadata={'help': 'Name of the BERT HuggingFace model to use.'})
    save_path: Path = field(metadata={'help': 'Trained model save path.'})
    dropout: float = field(default=0.5, metadata={'help': 'Dropout for BERT representations.'})


class NERModel(Module):

    def __init__(self, model_args: ModelArguments, category_mapping: Dict[str, int], label_mapping: Dict[str, int]):
        super().__init__()

        self._category_mapping = deepcopy(category_mapping)
        self._category_id_mapping = invert(self._category_mapping)

        self._label_mapping = deepcopy(label_mapping)
        self._label_id_mapping = invert(self._label_mapping)

        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_args.bert_model)
        self._encoder: PreTrainedModel = AutoModel.from_pretrained(model_args.bert_model)
        self._encoder.config.return_dict = True

        num_categories = len(self._category_mapping)
        num_labels = len(self._label_mapping)

        class ClassificationHead(Module):

            def __init__(self):
                super().__init__()
                self._attention = DotProductAttention()
                self._transition = Linear(728, num_labels)

            def forward(self, features: Tensor) -> Tensor:
                return self._transition(self._attention(features, features))

        self._dropout = Dropout(model_args.dropout)

        # create separate head for each category
        self._heads = ModuleList([ClassificationHead() for _ in range(num_categories)])

        self._optimized = False

    def train(self: _ModelType, mode: bool = True) -> _ModelType:
        if self._optimized:
            raise RuntimeError(f'{self.__class__.__name__} cannot be used in training after optimization!')
        return super(NERModel, self).train(mode)

    def predict(self, texts: List[str], *, batch_size: int = 1) -> List[Set[TypedSpan]]:
        encodings = self.tokenize(texts)

        # create collector for each category
        text_token_lengths = list(map(lambda encoding: len(encoding.ids), encodings))
        collectors = [PredictionsCollector(text_token_lengths) for _ in self._num]

        example_iterator = convert_all_to_examples(
            encodings, self._category_mapping, self._label_mapping,
            max_length=self._tokenizer.model_max_length
        )

        for batch in batch_examples(example_iterator, batch_size=batch_size):
            heads_logits = self(batch)
            predictions = list(map(to_label_ids, heads_logits))

            # iterate over batch
            for text_id, example_start, labels, mask in zip(batch.text_ids, batch.example_starts, predictions, batch.padding_mask):
                # iterate over categories
                for collector, head_prediction in zip(collectors, predictions):
                    collector.add(text_id, example_start, labels[mask])

        all_entities: List[Set[TypedSpan]] = [set() for _ in text_token_lengths]
        for category_id, collector in enumerate(collectors):
            for text_encoding, text_prediction, text_entities in zip(encodings, collector.predictions, all_entities):
                text_entities.update(decode(text_encoding, text_prediction, self._category_id_mapping[category_id]))

        return all_entities

    def tokenize(self, texts: List[str]) -> List[EncodingFast]:
        batch_encoding = self._tokenizer(texts, return_offsets_mapping=True, add_special_tokens=False)
        return batch_encoding.encodings

    def forward(self, examples: BatchedExamples) -> List[Tensor]:
        # TODO: fixme does not satisfy huggingface interface
        encoded: BaseModelOutput = self._encoder(input_ids=examples.input_ids, attention_mask=examples.padding_mask)
        representation = self._dropout(encoded['last_hidden_state'])

        return [head(representation) for head in self._heads]

    def optimize(self, onnx_dir: Path, fuse: bool = True, quant: bool = True) -> None:
        if self._optimized:
            raise RuntimeError(f'{self.__class__.__name__} has already been optimized!')
        self.eval()

        onnx_model_path = onnx_dir.joinpath('model.onnx')
        onnx_optimized_model_path = onnx_dir.joinpath('model-optimized.onnx')

        # load config
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(self._encoder)
        onnx_config: OnnxConfig = model_onnx_config(self._encoder.config)

        model_inputs = onnx_config.generate_dummy_inputs(self._tokenizer, framework=TensorType.PYTORCH)
        dynamic_axes = {0: 'batch', 1: 'sequence'}
        # export to onnx
        export(
            self._encoder,
            ({'input_ids': model_inputs['input_ids'], 'attention_mask': model_inputs['attention_mask']},),
            f=onnx_model_path.as_posix(),
            verbose=True,
            input_names=('input_ids', 'attention_mask'),
            output_names=('last_hidden_state',),
            dynamic_axes={'input_ids': dynamic_axes, 'attention_mask': dynamic_axes, 'last_hidden_state': dynamic_axes},
            do_constant_folding=True,
            use_external_data_format=onnx_config.use_external_data_format(self._encoder.num_parameters()),
            enable_onnx_checker=True,
            opset_version=11,
        )

        if fuse:
            opt_options = BertOptimizationOptions('bert')
            opt_options.enable_embed_layer_norm = False

            optimizer.optimize_model(
                str(onnx_model_path),
                'bert',
                num_heads=12,
                hidden_size=768,
                optimization_options=opt_options
            ).save_model_to_file(str(onnx_optimized_model_path))

            onnx_model_path = onnx_optimized_model_path

        if quant:
            onnx_model_path = quantize(onnx_model_path)

        self._encoder = ONNXOptimizedEncoder(onnx_model_path)

        self._optimized = True

    def save(self, save_path: Path) -> None:
        with open(save_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


class ONNXOptimizedEncoder(Module):

    def __init__(self, onnx_path: Path):
        super().__init__()
        self._onnx_path = onnx_path
        self._session: Optional[InferenceSession] = None

    def __getstate__(self):
        state = deepcopy(self.__dict__)
        state.pop('_session')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._session = None

    def _start_session(self) -> None:
        # Few properties that might have an impact on performances (provided by MS)
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load the model as a graph and prepare the CPU backend
        self._session = InferenceSession(self._onnx_path, options, providers=['CPUExecutionProvider'])
        self._session.disable_fallback()

    def forward(self, input_ids: LongTensor, padding_mask: BoolTensor) -> Tensor:
        if self._session is None:
            logger.info(f'Starting inference session for {self._onnx_path}.')
            self._start_session()

        # Run the model (None = get all the outputs)
        return self._session.run(None, {'input_ids': to_numpy(input_ids), 'attention_mask': to_numpy(padding_mask)})


def to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def to_label_ids(logits: Tensor) -> LongTensor:
    return torch.argmax(logits.detach().cpu(), dim=-1).long()
