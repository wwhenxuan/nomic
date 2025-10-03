import logging
import os
from typing import Optional, Union, List, Tuple


import torch
from torch import nn

from transformers import PreTrainedModel

from time_embeds.models.configuration_time_embeds import TimeEmbedsConfig


logger = logging.getLogger(__name__)


class TimeEmbedsPretrainedModel(PreTrainedModel):
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
    def from_pretrained(
        cls,
        model_name, config = None, *inputs, **kwargs
    ):
        """_summary_
        """
        
        # Instance the time series embeds model
        if config is None:
            config = cls.config_class.from_pretrained(model_name)
        
        # Create the Model based on the config
        model = cls(config, *inputs)
        

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