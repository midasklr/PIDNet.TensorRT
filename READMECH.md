# PIDNet.TensorRT
 [PIDNet](https://github.com/XuJiacong/PIDNet)TensorRT工程

### 1. 导出 *.wts

将本工程 custom.py. 替换原始pytorch代码中 `tools/`, 并且设置 `--w` 为 `True`用于导出wts模型:

```
python tools/custom.py
```

![](./images/frankfurt_000000_003025_leftImg8bit1024x1024.png)

以上为pytorch推理结果，导出的模型为`PIDNet.wts`.

#### 2. check TensorRT path in CMakeList.txt and wts path in pidnet.cpp

#### 3. Convert wts to TensorRT engine

```
mkdir build
cd build
cmake ..
make -j8
./main -s
```

如上将模型转为 `PIDNetS.engine`.

#### 4. Demo

在build目录下创建sample文件夹，并且将图片放入其中：

```
./main -d samples
```

![](./images/result_frankfurt_000000_003025_leftImg8bit_kLINEAR.png)

#### ToDo

- [x] cuda preprocess

- [x] PIDNet-S

- [ ] PIDNet-M/L...

  

#### 说明

1. 本工程仅转PIDNet-S模型，M和L模型类似。
2. TensorRT输入为1024x1024，通过对TensorRT精度矫正发现比2048x1024精度损失更小，实测训练时候使用1024x1024相比2048x1024损失不大（1%以内？）。





