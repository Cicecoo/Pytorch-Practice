import torch
import torch.nn as nn

conv = nn.Conv2d(3, 3, 3)
# 为什么参数是“输入通道数”和“输出通道数”？kernel的实际形状由此反推？

print(conv.weight.shape)

conv1 = nn.Conv2d(1, 6, 3)

print(conv1.weight.shape)

