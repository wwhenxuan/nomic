import logging
import os
from typing import Optional, Union, List, Tuple


import torch
from torch import nn
from torch.nn import init

from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

from time_embeds.models.configuration_time_embeds import TimeEmbedsConfig


logger = logging.getLogger(__name__)


@dataclass
class NomicBertMoEOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[List[torch.FloatTensor]] = None
    router_loss: Optional[torch.FloatTensor] = None
    tokens_per_expert: Optional[torch.LongTensor] = None


@dataclass
class TimeEmbedsOutput(ModelOutput):

    pass


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class TimeEmbedsPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    config_class = TimeEmbedsConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: TimeEmbedsConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 检查输入的配置文件是否正确
        if not isinstance(config, TimeEmbedsConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `TimeEmbedsConfig`!"
            )

        self.config = config

    @classmethod
    def from_pretrained(cls, model_name, config=None, *inputs, **kwargs):
        """_summary_"""

        # Instance the time series embeds model
        if config is None:
            config = cls.config_class.from_pretrained(model_name)

        # Create the Model based on the config
        model = cls(config, *inputs)

    def _init_weights(self, module: nn.Module) -> None:
        """
        初始化预训练网络模型的参数

        :param module: (nn.Module) The module in the pre-training neural network.
        :return: None
        """
        # 获取正态标准化的标准差信息
        initializer_range = self.config.initializer_range

        if isinstance(module, nn.Linear):
            init.normal_(module.weight, std=initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)

        elif isinstance(module, nn.Conv1d):
            init.normal_(module.weight, std=initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)


class TimeEmbedsForPreTraining(TimeEmbedsPreTrainedModel):
    """用于预训练的TimeEmbeds类"""

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

    def forward():
        """这里需要输出预训练的损失"""


# https://github.com/huggingface/transformers/blob/7032e0203262ebb2ebf55da8d2e01f873973e835/src/transformers/models/bert/modeling_bert.py#L748
def _init_weights(module, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])


class TimeEmbeds(TimeEmbedsPreTrainedModel):
    """_summary_

    Args:
        TimeEmbedsPretrainedModel (_type_): _description_
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)


class FineTuningModel(TimeEmbeds):
    """
    使用TimeEmbeds进行下游任务微调的模型
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
