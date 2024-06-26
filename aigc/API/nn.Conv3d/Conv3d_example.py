import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

# 输入的shape为（N，C，D，H，W）
x_ = np.ones((16, 3, 10, 32, 32)) 
x = Tensor(x_, mindspore.float32)

# nn.Conv3d前三个参数为必选项，且in_channels的值必须与输入的C相同
net = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(4, 3, 3), has_bias=True, pad_mode='valid') 

output = net(x)
output_shape = net(x).shape

print(output_shape)