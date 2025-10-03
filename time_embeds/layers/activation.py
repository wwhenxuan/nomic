import torch
from torch import nn

from typing import Union


ACTIVATION_DICT = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "sigmoid": nn.Sigmoid(),
    "logsigmoid": nn.LogSigmoid(),
}


def quick_gelu(input):
    return input * torch.sigmoid(1.702 * input)


def get_activation(activation: Union[nn.Module, str]) -> nn.Module:
    """
    根据输入的字符串来获取使用的激活函数。

    :param activation: Union(nn.Module, str) the name of the activation function.

    :return: (nn.Module) the selected activation function.
    """
    if callable(activation):
        # 判断输入是否是可直接调用的对象
        return activation()

    # 判断是否为字符串
    assert activation.lower() in ACTIVATION_DICT.keys()

    return ACTIVATION_DICT[activation]


if __name__ == "__main__":
    for key, value in ACTIVATION_DICT.items():
        print(get_activation(activation=key))
