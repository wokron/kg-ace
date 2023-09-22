## 配置环境
```sh
pip install -r requirements.txt
```

或
```sh
pip install flair
pip install pandas
```

## 数据预处理
使用 preprocess.py 将数据集处理成易于加载的格式，避免像 kg-bert 中那样每次训练都需要重新处理。
以 FB15k-237 为例，假设数据集位于 data/FB15k-237
```sh
python preprocess.py --data_dir data/FB15k-237
```
将生成文件 train_processed.csv、dev_processed.csv 和 test_processed.csv。

preprocess.py 中还有其他参数，可以指定要处理的数据集类型，正负例的比例等等。

## 分类器的训练
这里先实现了一个分类器，之后还需要添加 ace 的强化学习模型。可以先用数据集训练一下这个分类器，看看效果，之后再做剩下的内容。

使用 train.py 进行训练。
```sh
python train.py
```

这里也提供了一些配置的选项，可以通过命令行参数或 config.yaml 修改。

## 模型评价
使用 pandas 和 numpy 写了一个评价程序 eval.py，可以评价 hit@k、mr 和 mrr，但不知道效率如何，不过应该比纯 python 代码要快。同样有命令行参数可以选择。
```sh
python eval.py --data_dir data/FB15k-237 --model_path output/kg-ace/kg-ace-00020
```
