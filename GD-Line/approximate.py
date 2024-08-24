# 用梯度下降实现类似最小二乘法的拟合效果
# 算不算回归？
import torch

# 目标函数的构成：y=k*x+b，则对应 W = [k,b]
# 损失函数 l = y_-y = c-(k*x+b)
# 令 x 实现为 [x,1], 则 x*W^T = y
# [1., 1.] 相当于固定值初始化
w = torch.tensor([1., 1.], requires_grad=True)

# 由当前参数和样本自变量计算因变量对应前向
def forward(x):
    x_ = torch.tensor([x, 1.])
    # print(x_)
    # print(w)
    # print(x*w)
    # print(w.transpose(-1, 0))
    # print(x*w.transpose(-1, 0))
    # 直接 * 不是内积，是什么？ [1.,1.]*[0.,1.] = [0.,0.] ?
    return torch.dot(x_, w)

def loss(y, x):
    y_=forward(x)
    return (y_-y)**2

# 生成采样数据
x = torch.arange(0, 10, 0.1)
y_temp = [2*x+5 for x in x]
y = [torch.normal(y_t, 2) for y_t in y_temp]
print(x)
print(y)

# 学习率
lr = 0.001
for _ in range(1000):
    for i in range(len(x)):
        x_ = x[i]
        y_ = y[i]
        l = loss(y_, x_)
        l.backward()

        print('loss: ', l)
        print('grad: ', w.grad)

        # w = w - lr * w.grad
        # 直接修改w会创建新的张量？使之不再为叶子，梯度不被填充

        if w.grad is not None:
            # 更新参数
            with torch.no_grad(): # 使更新梯度时不记录计算图
                w -= lr * w.grad # -= 是inplace的 https://pytorch.org/docs/stable/generated/torch.Tensor.sub_.html
                # 还需要清除梯度
                w.grad.zero_() # 每次调用backward时梯度是积累的
                print('w: ', w)

print('iterated: ', w)

# 最小二乘法对比
x_mean = torch.mean(x)
y_mean = torch.mean(y)

x_e = x - torch.full_like(x, x_mean)
y_e = y - torch.full_like(y, y_mean)

a = torch.sum(x_e*y_e) / torch.sum(x_e*y_e)
b = y_mean - a * x_mean
print('leastsq: ', a, b)