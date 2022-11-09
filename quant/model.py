import logging
import pickle
from copy import deepcopy
from dataclasses import dataclass, field
from os import environ, cpu_count
from pathlib import Path
from typing import TypeVar, Tuple, List, Type, Optional, Set, Dict, Union

import torch
from torch import LongTensor, BoolTensor, Tensor
from torch.nn import Module, Dropout, Parameter, CrossEntropyLoss, Conv2d, LayerNorm
from torch.onnx import export
from transformers import AutoModel, AutoTokenizer, TensorType, PreTrainedTokenizer, PreTrainedModel
from transformers.convert_graph_to_onnx import quantize
from transformers.modeling_outputs import BaseModelOutput
from transformers.onnx import FeaturesManager, OnnxConfig
from transformers.tokenization_utils_base import EncodingFast

from quant.datamodel import TypedSpan, batch_examples, convert_all_to_examples, BatchedExamples
from quant.utils import invert, to_numpy, pad_images

# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
environ["OMP_NUM_THREADS"] = str(cpu_count())
environ["OMP_WAIT_POLICY"] = 'ACTIVE'

logger = logging.getLogger(__name__)

try:
    from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
    from onnxruntime_tools import optimizer
    from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions
except:
    logger.warning('Could not import ONNX inference tools!')

_ModelType = TypeVar('_ModelType', bound=Module)

logger = logging.getLogger(__name__)


class SerializableModel(Module):

    def __init__(self):
        super().__init__()
        self._dummy_param = Parameter(torch.empty(0))

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


@dataclass
class ModelArguments:
    bert_model: str = field(metadata={'help': 'Name of the BERT HuggingFace model to use.'})
    save_path: Path = field(metadata={'help': 'Trained model save path.'})
    dropout: float = field(default=0.5, metadata={'help': 'Dropout for BERT representations.'})


class SpanNERModel(SerializableModel):

    def __init__(self, model_args: ModelArguments, categories: Set[str], limit_entity_length: int = 1000):
        super().__init__()

        self._no_entity_category = 'NO_ENTITY'

        categories.add(self._no_entity_category)
        n_categories = len(categories)
        # noinspection PyTypeChecker
        self._category_id_mapping = dict(enumerate(categories))
        self._category_mapping = invert(self._category_id_mapping)
        self._no_entity_id = self._category_mapping[self._no_entity_category]

        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_args.bert_model)
        self._encoder: PreTrainedModel = AutoModel.from_pretrained(model_args.bert_model)
        self._context_length = self._encoder.config.max_position_embeddings

        self._dropout = Dropout(model_args.dropout)
        self._norm = LayerNorm(normalized_shape=[self._context_length, self._context_length])
        self._transition = Conv2d(in_channels=1, out_channels=n_categories, kernel_size=(3, 1), padding=(1, 0))

        positions = torch.arange(self._context_length)
        start_positions = positions.unsqueeze(-1).repeat(1, self._context_length)
        end_positions = positions.unsqueeze(-2).repeat(self._context_length, 1)
        self._spans = torch.cat([start_positions.unsqueeze(-1), end_positions.unsqueeze(-1)], dim=-1)  # (LENGTH, LENGTH, 2)

        entity_lengths = end_positions - start_positions + 1
        self._size_limit_mask = torch.triu((entity_lengths > 0) & (entity_lengths < limit_entity_length))

        self._optimized = False

    @property
    def category_mapping(self) -> Dict[str, int]:
        return deepcopy(self._category_mapping)

    @property
    def category_id_mapping(self) -> Dict[str, int]:
        return deepcopy(self._category_id_mapping)

    @property
    def no_entity_category(self) -> str:
        return self._no_entity_category

    @property
    def no_entity_category_id(self) -> int:
        return self._category_mapping[self._no_entity_category]

    @property
    def pad_token_id(self) -> int:
        return self._encoder.config.pad_token_id

    @property
    def context_length(self) -> int:
        return self._context_length

    def train(self: _ModelType, mode: bool = True) -> _ModelType:
        if self._optimized:
            raise RuntimeError(f'{self.__class__.__name__} cannot be used in training after optimization!')
        return super(SpanNERModel, self).train(mode)

    @torch.no_grad()
    def predict(self, texts: List[str], *, batch_size: int = 1) -> List[Set[TypedSpan]]:
        self.eval()
        encodings = self.tokenize(texts)

        example_iterator = convert_all_to_examples(
            encodings, self._category_mapping, self._label_mapping,
            max_length=self._tokenizer.model_max_length
        )

        all_entities: List[Set[TypedSpan]] = [set() for _ in texts]
        for batch, _ in batch_examples(example_iterator, batch_size=batch_size):
            predicted_category_ids = self(batch)
            _, length, _ = predicted_category_ids.shape

            entity_ids_mask = (predicted_category_ids != self._no_entity_category)

            start_padding_mask = batch.padding_mask.unsqueeze(-1)  # (BATCH, LENGTH, 1)
            end_padding_mask = batch.padding_mask.unsqueeze(-2)  # (BATCH, 1, LENGTH)

            final_mask = entity_ids_mask & self._size_limit_mask & start_padding_mask & end_padding_mask

            example_starts = torch.tensor(batch.example_starts).unsqueeze(-1)
            entity_spans = example_starts + self._spans.unsqueeze(0)  # example shift + relative shift

            entity_text_ids = torch.tensor(batch.text_ids).view(batch_size, 1, 1).repeat(1, length, length)

            chosen_text_ids = entity_text_ids[final_mask]
            chosen_spans = entity_spans[final_mask]
            chosen_category_ids = predicted_category_ids[final_mask]
            for text_id, category_id, (start, end) in zip(chosen_text_ids, chosen_category_ids, chosen_spans):
                all_entities[text_id].add(TypedSpan(start, end, self._category_id_mapping[category_id]))

        return all_entities

    def tokenize(self, texts: List[str]) -> List[EncodingFast]:
        batch_encoding = self._tokenizer(texts, return_offsets_mapping=True, add_special_tokens=False)
        return batch_encoding.encodings

    def forward(self, examples: BatchedExamples, labels: Optional[LongTensor] = None) -> Union[Tuple[Tensor, Tensor], Tensor]:
        encoded: BaseModelOutput = self._encoder(
            input_ids=examples.input_ids.to(self.device),
            attention_mask=examples.padding_mask.to(self.device)
        )
        representation: Tensor = self._dropout(encoded['last_hidden_state'])  # (B, L, F)

        category_scores = self._transition(representation.unsqueeze(1))  # (B, C, L, F)
        batch_size, num_categories, sequence_length, num_features = category_scores.shape
        logits = torch.bmm(
            category_scores.view(-1, sequence_length, num_features),
            category_scores.transpose(-2, -1).view(-1, num_features, sequence_length)
        ).view(batch_size, num_categories, sequence_length, sequence_length)  # convert to (B, C, L, L)

        logits = self._norm(
            pad_images(logits.view(-1, sequence_length, sequence_length), padding_length=self._context_length)
        ).view(batch_size, num_categories, self._context_length, self._context_length)

        logits = logits.transpose(-3, -2).transpose(-2, -1)  # (B, L, L, C)

        start_padding_mask = examples.padding_mask.unsqueeze(-2).to(self.device)
        end_padding_mask = examples.padding_mask.unsqueeze(-1).to(self.device)
        padding_image = pad_images(start_padding_mask & end_padding_mask, padding_length=self._context_length, padding_value=False)

        size_limit_mask = self._size_limit_mask.to(self.device)
        predictions_mask = size_limit_mask & padding_image

        predictions = torch.argmax(logits, dim=-1)
        predictions[~predictions_mask] = -100

        if labels is not None:
            labels = labels.to(self.device)
            labels_mask = size_limit_mask & (labels != -100)

            non_entity_mask = (labels != self._no_entity_id)

            loss = CrossEntropyLoss(reduction='mean')(logits[non_entity_mask], labels[non_entity_mask])
            return loss, predictions

        return predictions

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
