import torch
import torch.nn as nn

conv = nn.Conv2d(3, 3, 3)
# 为什么参数是“输入通道数”和“输出通道数”？kernel的实际形状由此反推？

print(conv.weight.shape)

conv1 = nn.Conv2d(1, 6, 3)

print(conv1.weight.shape)

''' output
torch.Size([3, 3, 3, 3])
torch.Size([6, 1, 3, 3])
'''

# The number of kernel groups is equal to the number of output feature maps.
# 所以实际不是由卷积核的通道数决定输出通道数
# 而是由卷积核的“组数” —— 将单通道的kernel视作“一个卷积核”，则三通道的kernel视作“一组三个卷积核”
# [6, 1, 3, 3]：六组，每组一个卷积核、卷积核大小3x3