### 简介

这个工程是我在学习**如何在PyTorch中用C++及CUDA自定义插件**时整理的内容。目前有两个算子, 一个是**GELU**, 参考了张校捷所著的《深入浅出PyTorch——从模型到源码》第八章的内容。

这个例子主要实现了一个叫做GELU的激活函数的前向和反传部分。GELU的表示式为: $GELU(x) = x * sigmoid(x)$。其导函数为: $GELU'(x) = sigmoid(1.702x) + 1.702x * sigmoid(1.702x)(1 - sigmoid(1.702x))$;

第二个例子是参考[这篇博客](https://zhuanlan.zhihu.com/p/350651297)，实现了一个称之为**NCReLU**的算子，其表达式为$NCReLU(x) = concat(ReLU(x), -RELU(-x))$。原谅我数学不好，反传的表示式写的不太对(不过不影响整体的训练流程了)，希望数学好的你能够帮我改一下。我在这篇博客的基础上主要又做了四件工作：1. 将CPU和GPU的代码合成一个算子；2. 添加了反传的逻辑; 3. 将这个算子应用到一个LeNet网络中进行训练；4. 将新的Op导出到onnx里面。

### 使用方法

- GELU

**编译**：

```
cd gelu
python3 setup.py install # 不想装到Python系统目录下可以使用python3 setup.py build_ext --inplace
```

**测试**：

```
# 前向传播
>>> import torch
>>> import gelu
>>> a = torch.randn(1, 3, 224, 224)
>>> a
tensor([[[[ 0.2852,  0.8364,  0.5310,  ...,  0.5861,  1.9312,  0.9830],
          [ 0.5820,  1.2994,  0.8673,  ..., -0.8407, -0.2846, -0.7699],
          [-2.7498, -0.6151, -1.3091,  ..., -0.8642, -0.3784, -1.5990],
          ...,
          [-0.0563, -0.7426,  1.7071,  ...,  0.1661,  1.1472,  1.6081],
          [ 0.3537, -0.7227, -0.2428,  ..., -1.3950,  0.2201, -0.1622],
          [-1.0310, -0.2678, -1.0553,  ..., -2.5736,  1.0485, -0.2778]],

         [[ 1.8638, -0.0933, -1.2488,  ..., -1.0449,  0.4978, -1.4200],
          [ 0.4051,  0.2335,  0.3398,  ...,  1.4184, -0.2767, -0.2732],
          [ 0.3788, -0.0167, -0.8282,  ...,  0.2951, -0.0259,  0.2375],
          ...,
          [ 0.4036, -0.6680, -0.1017,  ..., -1.2147,  1.5642,  0.0466],
          [-0.2857,  0.4009,  0.3462,  ...,  1.2096,  1.1008, -1.1620],
          [ 0.9911,  1.3641, -0.0114,  ..., -0.7196,  0.7288, -0.4400]],

         [[-0.2088, -0.6336,  1.6053,  ..., -1.0999, -0.9074, -0.1060],
          [ 0.1978, -1.2500,  0.3350,  ...,  0.0659,  0.5051, -0.6331],
          [ 1.6607, -0.7336, -1.4014,  ...,  0.7068,  0.4102,  0.1277],
          ...,
          [-0.0693, -1.6408,  1.0105,  ..., -0.6562,  1.3368, -1.3679],
          [ 1.0656,  1.4306, -0.4094,  ..., -0.7474, -0.6004, -0.2139],
          [-2.6281,  1.8857,  0.5881,  ...,  0.3812,  0.2241,  1.2721]]]])
>>> b = gelu.forward(a)
>>> b
tensor([[[[ 0.1766,  0.6740,  0.3779,  ...,  0.4282,  1.8616,  0.8276],
          [ 0.4244,  1.1711,  0.7059,  ..., -0.1622, -0.1085, -0.1635],
          [-0.0253, -0.1598, -0.1273,  ..., -0.1614, -0.1303, -0.0987],
          ...,
          [-0.0268, -0.1636,  1.6185,  ...,  0.0947,  1.0047,  1.5103],
          [ 0.2286, -0.1635, -0.0967,  ..., -0.1188,  0.1304, -0.0700],
          [-0.1520, -0.1039, -0.1502,  ..., -0.0318,  0.8978, -0.1067]],

         [[ 1.7888, -0.0430, -0.1332,  ..., -0.1510,  0.3485, -0.1163],
          [ 0.2697,  0.1396,  0.2177,  ...,  1.3020, -0.1064, -0.1054],
          [ 0.2484, -0.0082, -0.1626,  ...,  0.1839, -0.0127,  0.1424],
          ...,
          [ 0.2685, -0.1622, -0.0465,  ..., -0.1364,  1.4622,  0.0242],
          [-0.1088,  0.2663,  0.2226,  ...,  1.0727,  0.9542, -0.1413],
          [ 0.8364,  1.2422, -0.0056,  ..., -0.1634,  0.5653, -0.1413]],

         [[-0.0860, -0.1608,  1.5072,  ..., -0.1466, -0.1596, -0.0482],
          [ 0.1154, -0.1331,  0.2140,  ...,  0.0348,  0.3548, -0.1608],
          [ 1.5679, -0.1636, -0.1182,  ...,  0.5436,  0.2739,  0.0708],
          ...,
          [-0.0326, -0.0947,  0.8571,  ..., -0.1618,  1.2123, -0.1215],
          [ 0.9163,  1.3153, -0.1361,  ..., -0.1636, -0.1589, -0.0877],
          [-0.0297,  1.8125,  0.4300,  ...,  0.2503,  0.1332,  1.1412]]]])
>>> a = a.to('cuda:0')
>>> b = gelu.forward(a)
>>> b
tensor([[[[ 0.1766,  0.6740,  0.3779,  ...,  0.4282,  1.8616,  0.8276],
          [ 0.4244,  1.1711,  0.7059,  ..., -0.1622, -0.1085, -0.1635],
          [-0.0253, -0.1598, -0.1273,  ..., -0.1614, -0.1303, -0.0987],
          ...,
          [-0.0268, -0.1636,  1.6185,  ...,  0.0947,  1.0047,  1.5103],
          [ 0.2286, -0.1635, -0.0967,  ..., -0.1188,  0.1304, -0.0700],
          [-0.1520, -0.1039, -0.1502,  ..., -0.0318,  0.8978, -0.1067]],

         [[ 1.7888, -0.0430, -0.1332,  ..., -0.1510,  0.3485, -0.1163],
          [ 0.2697,  0.1396,  0.2177,  ...,  1.3020, -0.1064, -0.1054],
          [ 0.2484, -0.0082, -0.1626,  ...,  0.1839, -0.0127,  0.1424],
          ...,
          [ 0.2685, -0.1622, -0.0465,  ..., -0.1364,  1.4622,  0.0242],
          [-0.1088,  0.2663,  0.2226,  ...,  1.0727,  0.9542, -0.1413],
          [ 0.8364,  1.2422, -0.0056,  ..., -0.1634,  0.5653, -0.1413]],

         [[-0.0860, -0.1608,  1.5072,  ..., -0.1466, -0.1596, -0.0482],
          [ 0.1154, -0.1331,  0.2140,  ...,  0.0348,  0.3548, -0.1608],
          [ 1.5679, -0.1636, -0.1182,  ...,  0.5436,  0.2739,  0.0708],
          ...,
          [-0.0326, -0.0947,  0.8571,  ..., -0.1618,  1.2123, -0.1215],
          [ 0.9163,  1.3153, -0.1361,  ..., -0.1636, -0.1589, -0.0877],
          [-0.0297,  1.8125,  0.4300,  ...,  0.2503,  0.1332,  1.1412]]]],
       device='cuda:0')
```

```
# 反向传播
>>> import torch
>>> import gelu
>>> a = torch.randn(1, 3, 224, 224, requires_grad=True)
>>> b = gelu.forward(a)
>>> c = b.sum()
>>> a
tensor([[[[-1.2359e-01,  2.9992e-01,  1.2691e-01,  ...,  9.2793e-01,
            2.8796e-01, -3.6482e-03],
          [-8.4472e-01,  1.3253e-01, -1.7693e+00,  ...,  2.0399e+00,
           -5.1578e-01, -8.8441e-01],
          [ 3.1571e+00, -1.0266e+00,  4.3880e-01,  ...,  1.5795e+00,
            1.0834e+00,  2.2546e+00],
          ...,
          [ 1.2417e+00,  1.1198e+00, -1.1429e+00,  ..., -7.8511e-01,
            8.4822e-01, -4.9575e-01],
          [ 4.5461e-02,  1.8928e-02,  9.4056e-01,  ...,  1.1566e+00,
           -1.4095e+00,  3.1579e-01],
          [ 9.3418e-01, -3.4380e-01, -1.6648e-01,  ..., -1.3711e+00,
           -1.2797e+00, -3.3876e-03]],

         [[-2.7568e-01,  1.6503e+00,  8.1311e-01,  ...,  5.4260e-01,
            3.1233e-01, -7.0955e-02],
          [ 1.1402e+00, -3.6503e-01,  9.7429e-01,  ...,  7.2664e-02,
            4.9484e-01, -2.7287e-02],
          [ 1.1161e-01, -8.7284e-01, -7.6045e-01,  ..., -2.5578e-02,
           -1.1839e+00,  8.6934e-01],
          ...,
          [-2.4249e-01, -3.0164e-01,  2.5160e+00,  ..., -1.6725e-01,
           -2.8124e-03,  8.1270e-01],
          [ 1.3385e+00, -4.4661e-01, -4.4738e-01,  ..., -8.0067e-01,
            1.2323e-01, -3.6359e-01],
          [ 1.2001e+00,  2.6245e+00,  9.4692e-01,  ..., -3.1271e-01,
            9.3609e-01,  3.2724e-01]],

         [[-4.2722e-01,  2.0683e-02,  3.0053e-01,  ...,  8.7674e-01,
           -6.6023e-01, -1.8551e-01],
          [-8.2130e-01,  6.0544e-01,  1.7471e-02,  ..., -2.2975e-01,
            1.1338e+00, -1.1824e+00],
          [ 1.0890e+00,  1.0786e+00,  8.4522e-01,  ..., -1.0034e+00,
           -3.7159e-01, -1.0198e+00],
          ...,
          [ 1.4833e-02,  6.3037e-02,  3.3636e-01,  ..., -2.8834e+00,
            4.5519e-01, -3.6897e-01],
          [ 1.8372e-01,  3.4452e-01,  4.7947e-01,  ...,  2.3552e+00,
           -2.6487e-01,  3.6537e-01],
          [-8.4903e-01, -1.5825e-01, -1.0737e+00,  ..., -1.0595e+00,
           -1.0536e-01,  2.9452e-01]]]], requires_grad=True)
>>> b
tensor([[[[-5.5318e-02,  1.8742e-01,  7.0279e-02,  ...,  7.6936e-01,
            1.7857e-01, -1.8184e-03],
          [-1.6210e-01,  7.3705e-02, -8.3007e-02,  ...,  1.9785e+00,
           -1.5144e-01, -1.6065e-01],
          [ 3.1425e+00, -1.5234e-01,  2.9772e-01,  ...,  1.4789e+00,
            9.3537e-01,  2.2070e+00],
          ...,
          [ 1.1078e+00,  9.7487e-01, -1.4295e-01,  ..., -1.6340e-01,
            6.8623e-01, -1.4909e-01],
          [ 2.3609e-02,  9.6166e-03,  7.8267e-01,  ...,  1.0149e+00,
           -1.1734e-01,  1.9933e-01],
          [ 7.7594e-01, -1.2299e-01, -7.1526e-02,  ..., -1.2118e-01,
           -1.3020e-01, -1.6889e-03]],

         [[-1.0608e-01,  1.5565e+00,  6.5018e-01,  ...,  3.8837e-01,
            1.9673e-01, -3.3338e-02],
          [ 9.9702e-01, -1.2757e-01,  8.1840e-01,  ...,  3.8576e-02,
            3.4586e-01, -1.3327e-02],
          [ 6.1089e-02, -1.6112e-01, -1.6359e-01,  ..., -1.2511e-02,
           -1.3927e-01,  7.0809e-01],
          ...,
          [-9.6573e-02, -1.1293e-01,  2.4818e+00,  ..., -7.1801e-02,
           -1.4028e-03,  6.4976e-01],
          [ 1.2141e+00, -1.4230e-01, -1.4242e-01,  ..., -1.6317e-01,
            6.8052e-02, -1.2727e-01],
          [ 1.0623e+00,  2.5947e+00,  7.8939e-01,  ..., -1.1570e-01,
            7.7795e-01,  2.0804e-01]],

         [[-1.3920e-01,  1.0523e-02,  1.8788e-01,  ...,  7.1578e-01,
           -1.6197e-01, -7.8232e-02],
          [-1.6275e-01,  4.4621e-01,  8.8652e-03,  ..., -9.2695e-02,
            9.9002e-01, -1.3940e-01],
          [ 9.4152e-01,  9.3025e-01,  6.8314e-01,  ..., -1.5397e-01,
           -1.2893e-01, -1.5283e-01],
          ...,
          [ 7.5102e-03,  3.3208e-02,  2.1504e-01,  ..., -2.1153e-02,
            3.1160e-01, -1.2839e-01],
          [ 1.0610e-01,  2.2137e-01,  3.3246e-01,  ...,  2.3132e+00,
           -1.0308e-01,  2.3772e-01],
          [-1.6196e-01, -6.8534e-02, -1.4875e-01,  ..., -1.4987e-01,
           -4.7969e-02,  1.8342e-01]]]], grad_fn=<MulBackward0>)
>>> c
tensor(42428.2500, grad_fn=<SumBackward0>)
>>> a.grad
>>> c.backward()
>>> a.grad
tensor([[[[ 0.3956,  0.7446,  0.6072,  ...,  1.0529,  0.7356,  0.4969],
          [-0.0311,  0.6118, -0.0877,  ...,  1.0713,  0.1115, -0.0421],
          [ 1.0201, -0.0724,  0.8414,  ...,  1.0966,  1.0809,  1.0581],
          ...,
          [ 1.0955,  1.0853, -0.0878,  ..., -0.0121,  1.0321,  0.1233],
          [ 0.5386,  0.5161,  1.0557,  ...,  1.0891, -0.0998,  0.7563],
          [ 1.0543,  0.2233,  0.3602,  ..., -0.0996, -0.0973,  0.4971]],

         [[ 0.2737,  1.0938,  1.0214,  ...,  0.9036,  0.7538,  0.4398],
          [ 1.0875,  0.2082,  1.0629,  ...,  0.5617,  0.8762,  0.4768],
          [ 0.5944, -0.0390, -0.0034,  ...,  0.4782, -0.0915,  1.0381],
          ...,
          [ 0.2994,  0.2541,  1.0439,  ...,  0.3596,  0.4976,  1.0212],
          [ 1.0991,  0.1536,  0.1531,  ..., -0.0173,  0.6041,  0.2093],
          [ 1.0928,  1.0388,  1.0572,  ...,  0.2459,  1.0547,  0.7647]],

         [[ 0.1661,  0.5176,  0.7450,  ...,  1.0401,  0.0373,  0.3447],
          [-0.0239,  0.9367,  0.5149,  ...,  0.3094,  1.0869, -0.0914],
          [ 1.0816,  1.0802,  1.0312,  ..., -0.0684,  0.2037, -0.0713],
          ...,
          [ 0.5126,  0.5535,  0.7713,  ..., -0.0284,  0.8518,  0.2055],
          [ 0.6538,  0.7772,  0.8669,  ...,  1.0524,  0.2820,  0.7920],
          [-0.0323,  0.3669, -0.0796,  ..., -0.0775,  0.4108,  0.7405]]]])
```

- NCReLU

**编译**

```
cd ncrelu
python3 setup.py install
```

**训练**

```
python3 train.py
```

**测试及onnx导出**

```
python3 predict.py
```

