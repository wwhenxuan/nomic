import torch
from torch import nn

from time_embeds.layers.activation import get_activation

from typing import Optional


class TimeEmbeddings(nn.Module):
    """
    以通道独立的方式对输入的时间序列数据进行嵌入。
    input: [batch_size * num_vars, num_tokens, patch_len],
    output: [batch_size * num_vars, num_tokens, d_model].

    这里提供了直接嵌入和残差嵌入两种不同的嵌入形式。
    """

    def __init__(
        self,
        patch_len: int,
        d_model: int,
        bias: Optional[bool] = True,
        residual_embedding: Optional[bool] = True,
        hidden_features: Optional[int] = None,
        activation: Optional[str] = "relu",
    ) -> None:
        """
        :param patch_len: (int) the length of the patch of time series.
        :param d_model: (int) the dimension of the embedding model.
        :param bias: (bool) whether to open the bias for linear mapping.
        :param residual_embedding: (bool) whether to use the residual embedding.
        :param hidden_features: (int) the dimension of the hidden embedding layers.
        :param activation: (str) the activation function for the residual embeddings.
        """
        super(self, TimeEmbeddings).__init__()

        self.patch_len = patch_len

        self.hidden_features = (
            hidden_features if hidden_features is not None else d_model // 2
        )

        # 是否要通过残差连接的方式进行时间序列的嵌入
        self.residual_embedding = residual_embedding

        if self.residual_embedding is True:
            # 通过残差连接的方式进行时间序列的嵌入
            self.hidden_layer = nn.Linear(
                in_features=patch_len, out_features=hidden_features, bias=bias
            )
            self.output_layer = nn.Linear(
                in_features=hidden_features, out_features=d_model, bias=bias
            )
            self.residual_layer = nn.Linear(
                in_features=patch_len, out_features=d_model, bias=bias
            )

            self.activation = get_activation(activation=activation)

        else:
            self.linear_embedding = nn.Linear(
                in_features=patch_len, out_features=d_model, bias=bias
            )

    def forward(
        self,
        time_series: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param time_series: (Tensor) the input time series with shape of
               [batch_size * num_vars, num_tokens, patch_len];

        :return: embeddings (Tensor) the output embedding of the time series data with shape of
                [batch * num_vars, num_tokens, d_model]
        """
        if self.residual_embedding is True:
            # 计算隐藏层特征
            hid = self.activation(self.hidden_layer(time_series))
            out = self.output_layer(hid)

            # 建立残差连接关系
            res = self.residual_layer(time_series)
            embeddings = out + res

        else:
            embeddings = self.linear_embedding(time_series)

        return embeddings
