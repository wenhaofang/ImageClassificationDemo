## Image Classification Demo

This repository includes some demo image classification models.

Note: The project refers to https://github.com/bentrevett/pytorch-image-classification

<br/>

Datasets

* `dataset1`: [MNIST](http://yann.lecun.com/exdb/mnist/)
* `dataset1`: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

Models

* `model1`: Multilayer Perceptron
* `model2`: LeNet
* `model3`: AlexNet

### Data Process

```shell
# process1
PYTHONPATH=. python dataprocess/process1.py \
    --loader_config_path configs/loader1.yaml
# process2
PYTHONPATH=. python dataprocess/process2.py \
    --loader_config_path configs/loader2.yaml
```

### Unit Test

* for loader

```shell
# loader1
PYTHONPATH=. python loaders/loader1.py \
    --loader_config_path configs/loader1.yaml
# loader2
PYTHONPATH=. python loaders/loader2.py \
    --loader_config_path configs/loader2.yaml
```

* for module

```shell
# module1
PYTHONPATH=. python modules/module1.py \
    --module_config_path configs/module1.yaml
# module2
PYTHONPATH=. python modules/module2.py \
    --module_config_path configs/module2.yaml
# module3
PYTHONPATH=. python modules/module3.py \
    --module_config_path configs/module3.yaml
```

### Main Process

```shell
python main.py
```

You can change the config either in the command line or in the file `utils/parser.py` and `configs/*.yaml`

Here are the examples for each module:

```shell
# loader1 & module1: MNIST & MLP
python main.py \
    --loader 1 \
    --loader_config_path configs/loader1.yaml \
    --module 1 \
    --module_config_path configs/module1.yaml \
    --name 1_1
```

```shell
# loader1 & module2: MNIST & LeNet
python main.py \
    --loader 1 \
    --loader_config_path configs/loader1.yaml \
    --module 2 \
    --module_config_path configs/module2.yaml \
    --name 1_2
```

```shell
# loader2 & module3: CIFAR-10 % AlexNet
python main.py \
    --loader 2 \
    --loader_config_path configs/loader2.yaml \
    --module 3 \
    --module_config_path configs/module3.yaml \
    --name 2_3 \
    --num_epochs 25
```
