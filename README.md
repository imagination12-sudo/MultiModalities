## 一、 训练代码启动
1. 预训练模型地址修改
在 ./experiments/untrack/deep_rgbx.yaml 下我标注了明显 “==” 注释的地方修改，并且标注了模型的名称，添加模型的**绝对路径**

2. 预训练模型均放在 ./pretrained 下

3. 文件保存路径
./lib/train/admin/local.py 下修改日志及数据集加载路径，相关修改在 local.py 下均有注释

4. 数据及序列名称
如果有很多序列需要加载训练，将其放在 ./lib/train/data_specs 下，训练集命名为 TrainSet_list.txt, 验证集命名为 ValidationSet_list.txt, 格式可以参考我放在此目录下的文件

5. 上述配置完成之后，运行./train.py文件


## 二、 测试代码启动
1. 训练模型权重
需放在 ./pretrained/models 下，命名为 UnTrack.pth.tar, 我已放在此目录下，一般情况不需要修改

2. 测试日志及测试集路径
需修改 ./lib/text/evaluation/local.py 文件，相关修改在 local.py 中均有注释

3. 配置完成后运行 ./inference.py 文件启用测试

## 三、 模型结构
1. 模型架构
模型架构主要放在 ./lib/models 下

2. 数据加载主要在 ./lib/train/data、./lib/train/dataset、./lib/train/data_specs下

3. 训练流程基本逻辑放在 ./lib/train/actors 下


## 四、 超参数
超参数均在 ./experiments/untrack/deep_rgbx.yaml 中