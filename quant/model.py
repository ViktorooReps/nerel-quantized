import logging
import pickle
import time
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from os import environ, cpu_count
from pathlib import Path
from typing import TypeVar, Tuple, List, Type, Optional, Set, Dict, Union

import torch
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions
from torch import LongTensor, BoolTensor, Tensor
from torch.nn import Module, Dropout, Parameter, CrossEntropyLoss, Linear, Bilinear
from torch.nn.functional import pad
from torch.onnx import export
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, TensorType, PreTrainedTokenizer, PreTrainedModel
from transformers.convert_graph_to_onnx import quantize
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.lxmert.modeling_lxmert import GeLU
from transformers.onnx import FeaturesManager, OnnxConfig
from transformers.tokenization_utils_base import EncodingFast

from quant.datamodel import TypedSpan, batch_examples, convert_all_to_examples, BatchedExamples, read_nerel, DatasetType, collate_examples
from quant.pruner import mask_heads, prune_heads
from quant.utils import invert, to_numpy, pad_images

torch.set_num_threads(cpu_count() // 2)

# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
environ["OMP_NUM_THREADS"] = str(cpu_count() // 2)
environ["OMP_WAIT_POLICY"] = 'ACTIVE'

logger = logging.getLogger(__name__)

try:
    from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
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
    reduced_dim: int = field(default=128, metadata={'help': 'Reduced token representation.'})
    max_context_length: int = field(default=None, metadata={'help': 'Context length (same as model by default)'})


class SpanNERModel(SerializableModel):

    def __init__(self, model_args: ModelArguments, categories: Set[str], limit_entity_length: int = 20):
        super().__init__()

        self._no_entity_category = 'NO_ENTITY'

        categories.add(self._no_entity_category)
        self._n_categories = len(categories)
        # noinspection PyTypeChecker
        self._category_id_mapping = dict(enumerate(categories))
        self._category_mapping = invert(self._category_id_mapping)
        self._no_entity_id = self._category_mapping[self._no_entity_category]

        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_args.bert_model)
        self._encoder: PreTrainedModel = AutoModel.from_pretrained(model_args.bert_model)

        self._context_length = self._encoder.config.max_position_embeddings
        if model_args.max_context_length is not None:
            self._context_length = model_args.max_context_length
        n_features = self._encoder.config.hidden_size

        self._dropout = Dropout(model_args.dropout)
        self._start_projection = Linear(n_features, model_args.reduced_dim)
        self._end_projection = Linear(n_features, model_args.reduced_dim)
        self._activation = GeLU()

        self._transition = Bilinear(model_args.reduced_dim, model_args.reduced_dim, self._n_categories)

        positions = torch.arange(self._context_length)
        start_positions = positions.unsqueeze(-1).repeat(1, self._context_length)
        end_positions = positions.unsqueeze(-2).repeat(self._context_length, 1)
        entity_lengths = end_positions - start_positions + 1
        self._size_limit_mask = torch.triu((entity_lengths > 0) & (entity_lengths < limit_entity_length))

        self._optimized = False

    @property
    def category_mapping(self) -> Dict[str, int]:
        return deepcopy(self._category_mapping)

    @property
    def category_id_mapping(self) -> Dict[int, str]:
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
        if self._optimized and mode:
            raise RuntimeError(f'{self.__class__.__name__} cannot be used in training after optimization!')
        return super(SpanNERModel, self).train(mode)

    @torch.no_grad()
    def predict(self, texts: List[str], *, batch_size: int = 1) -> List[Set[TypedSpan]]:
        self.eval()
        encodings = self.tokenize(texts)

        example_iterator = convert_all_to_examples(
            encodings,
            self._category_mapping,
            self.no_entity_category,
            max_length=self._context_length
        )

        all_entities: List[Set[TypedSpan]] = [set() for _ in texts]
        for batch in batch_examples(example_iterator, batch_size=batch_size):
            examples: BatchedExamples = batch['examples']
            predicted_category_ids: LongTensor = self(examples).cpu()
            _, length = examples.padding_mask.shape

            entity_ids_mask = (predicted_category_ids != self._no_entity_id)

            start_padding_mask = examples.padding_mask.unsqueeze(-1)  # (BATCH, LENGTH, 1)
            end_padding_mask = examples.padding_mask.unsqueeze(-2)  # (BATCH, 1, LENGTH)
            padding_image = pad_images(start_padding_mask & end_padding_mask, padding_length=self._context_length, padding_value=False)

            final_mask = entity_ids_mask.cpu() & self._size_limit_mask & padding_image

            entity_text_ids = torch.tensor(examples.text_ids).view(batch_size, 1, 1).repeat(1, self._context_length, self._context_length)

            chosen_spans = torch.cat([
                pad(
                    examples.start_offset,
                    [0, self._context_length - length],
                    value=-100
                ).view(batch_size, self._context_length, 1, 1).repeat(1, 1, self._context_length, 1),
                pad(
                    examples.end_offset,
                    [0, self._context_length - length],
                    value=-100
                ).view(batch_size, self._context_length, 1, 1).transpose(-3, -2).repeat(1, self._context_length, 1, 1)
            ], dim=-1)[final_mask]

            chosen_text_ids = entity_text_ids[final_mask]
            chosen_category_ids = predicted_category_ids[final_mask]
            for text_id, category_id, (start, end) in zip(chosen_text_ids, chosen_category_ids, chosen_spans):
                all_entities[text_id].add(TypedSpan(start.item(), end.item(), self._category_id_mapping[category_id.item()]))

        return all_entities

    def tokenize(self, texts: List[str]) -> List[EncodingFast]:
        batch_encoding = self._tokenizer(texts, return_offsets_mapping=True, add_special_tokens=False)
        return batch_encoding.encodings

    def forward(
            self,
            examples: BatchedExamples,
            labels: Optional[LongTensor] = None,
            encoder_head_mask: BoolTensor = None,
            return_attention_scores: bool = False
    ) -> Union[Tuple[Tensor, ...], Tensor]:

        if return_attention_scores and self._optimized:
            raise NotImplementedError

        encoded: BaseModelOutput = self._encoder(
            input_ids=examples.input_ids.to(self.device),
            attention_mask=examples.padding_mask.to(self.device),
            head_mask=encoder_head_mask
        )
        representation: Tensor = encoded['last_hidden_state']  # (B, L, F)
        attention_scores: Optional[Tensor] = encoded['attentions'] if return_attention_scores else None

        batch_size, sequence_length, num_features = representation.shape

        representation = pad(representation, [0, 0, 0, self._context_length - sequence_length, 0, 0])  # (B, M, F)

        start_representation = self._dropout(self._activation(self._start_projection(representation))).unsqueeze(-2)  # (B, M, 1, R)
        end_representation = self._dropout(self._activation(self._end_projection(representation))).unsqueeze(-3)  # (B, 1, M, R)

        category_scores = self._transition(
            start_representation.repeat(1, 1, self._context_length, 1),
            end_representation.repeat(1, self._context_length, 1, 1)
        )  # (B, M, M, C)

        start_padding_mask = examples.padding_mask.unsqueeze(-2).to(self.device)
        end_padding_mask = examples.padding_mask.unsqueeze(-1).to(self.device)
        padding_image = pad_images(start_padding_mask & end_padding_mask, padding_length=self._context_length, padding_value=False)

        size_limit_mask = self._size_limit_mask.to(self.device)
        predictions_mask = size_limit_mask & padding_image

        predictions = torch.argmax(category_scores, dim=-1)
        predictions[~predictions_mask] = -100

        if labels is not None:
            labels = labels.to(self.device)
            labels_mask = size_limit_mask & (labels != -100)

            entity_labels_mask = labels_mask & (labels != self._no_entity_id)
            entity_predictions_mask = predictions_mask & (predictions != self._no_entity_id)

            recall_loss = CrossEntropyLoss(reduction='mean')(category_scores[entity_labels_mask], labels[entity_labels_mask])
            precision_loss = CrossEntropyLoss(reduction='mean')(category_scores[entity_predictions_mask], labels[entity_predictions_mask])
            return (recall_loss + precision_loss, predictions) + ((attention_scores,) if return_attention_scores else tuple())

        if return_attention_scores:
            return predictions, attention_scores
        return predictions

    def prune(self, dataset_dir: Path, prune_fraction: float = 0.1, prune_iter: int = 5, batch_size: int = 16):
        if self._optimized:
            raise RuntimeError(f'{self.__class__.__name__} has already been optimized!')
        self.eval()

        from train import NERDataset

        dev_dataset = NERDataset(read_nerel(
            dataset_dir, DatasetType.DEV, self.tokenize, self._category_mapping, self._no_entity_category
        ))
        dataloader = DataLoader(
            dev_dataset,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=partial(collate_examples, pad_length=self._context_length)
        )
        head_mask = mask_heads(self, dataloader, prune_fraction=prune_fraction, num_iter=prune_iter)
        prune_heads(self, head_mask)

    def optimize(
            self,
            onnx_dir: Path,
            fuse: bool = True,
            quant: bool = True,
            opset_version: int = 13,
            do_constant_folding: bool = True
    ) -> None:

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
            verbose=False,
            input_names=('input_ids', 'attention_mask'),
            output_names=('last_hidden_state',),
            dynamic_axes={'input_ids': dynamic_axes, 'attention_mask': dynamic_axes, 'last_hidden_state': dynamic_axes},
            do_constant_folding=do_constant_folding,
            opset_version=opset_version,
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
        self._session = InferenceSession(self._onnx_path.as_posix(), options, providers=['CPUExecutionProvider'])
        self._session.disable_fallback()

    def forward(self, input_ids: LongTensor, attention_mask: BoolTensor, **_) -> Dict[str, Tensor]:
        if self._session is None:
            logger.info(f'Starting inference session for {self._onnx_path}.')
            start_time = time.time()
            self._start_session()
            logger.info(f'Inference started in {time.time() - start_time:.4f}s.')

        # Run the model (None = get all the outputs)
        return {
            'last_hidden_state': torch.tensor(self._session.run(
                None,
                {
                    'input_ids': to_numpy(input_ids),
                    'attention_mask': to_numpy(attention_mask.long())
                }
            )[0])
        }
