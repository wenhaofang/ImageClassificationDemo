## Image Classification Demo

This repository includes some demo image classification models.

Note: The project refers to https://github.com/bentrevett/pytorch-image-classification

<br/>

Datasets

* `dataset1`: [MNIST](http://yann.lecun.com/exdb/mnist/)
* `dataset2`: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

Models

* `model1`: Multilayer Perceptron
* `model2`: LeNet
* `model3`: AlexNet
* `model4`: VGG
* `model5`: ResNet

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
# loader1: MNIST
PYTHONPATH=. python loaders/loader1.py \
    --loader_config_path configs/loader1.yaml
# loader2: CIFAR-10
# You can add --use_pretrain to load dataset for pre-trained model
PYTHONPATH=. python loaders/loader2.py \
    --loader_config_path configs/loader2.yaml
```

* for module

```shell
# module1: MLP
PYTHONPATH=. python modules/module1.py \
    --loader_config_path configs/loader1.yaml \
    --module_config_path configs/module1.yaml
# module2: LeNet
PYTHONPATH=. python modules/module2.py \
    --loader_config_path configs/loader1.yaml \
    --module_config_path configs/module2.yaml
# module3: AlexNet
PYTHONPATH=. python modules/module3.py \
    --loader_config_path configs/loader2.yaml \
    --module_config_path configs/module3.yaml
# module4: vgg11
PYTHONPATH=. python modules/module4.py \
    --loader_config_path configs/loader2.yaml \
    --module_config_path configs/module4.yaml \
    --vgg_net_arch 11
# module4: vgg13
PYTHONPATH=. python modules/module4.py \
    --loader_config_path configs/loader2.yaml \
    --module_config_path configs/module4.yaml \
    --vgg_net_arch 13
# module4: vgg16
PYTHONPATH=. python modules/module4.py \
    --loader_config_path configs/loader2.yaml \
    --module_config_path configs/module4.yaml \
    --vgg_net_arch 16
# module4: vgg19
PYTHONPATH=. python modules/module4.py \
    --loader_config_path configs/loader2.yaml \
    --module_config_path configs/module4.yaml \
    --vgg_net_arch 19
# module5: resnet18
PYTHONPATH=. python modules/module5.py \
    --loader_config_path configs/loader2.yaml \
    --module_config_path configs/module5.yaml \
    --res_net_arch 18
# module5: resnet34
PYTHONPATH=. python modules/module5.py \
    --loader_config_path configs/loader2.yaml \
    --module_config_path configs/module5.yaml \
    --res_net_arch 34
# module5: resnet50
PYTHONPATH=. python modules/module5.py \
    --loader_config_path configs/loader2.yaml \
    --module_config_path configs/module5.yaml \
    --res_net_arch 50
# module5: resnet101
PYTHONPATH=. python modules/module5.py \
    --loader_config_path configs/loader2.yaml \
    --module_config_path configs/module5.yaml \
    --res_net_arch 101
# module5: resnet152
PYTHONPATH=. python modules/module5.py \
    --loader_config_path configs/loader2.yaml \
    --module_config_path configs/module5.yaml \
    --res_net_arch 152
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
# loader2 & module3: CIFAR-10 & AlexNet
python main.py \
    --loader 2 \
    --loader_config_path configs/loader2.yaml \
    --module 3 \
    --module_config_path configs/module3.yaml \
    --name 2_3 \
    --num_epochs 25
```

```shell
# loader2 & module4: CIFAR-10 & VGG
# You can add --use_pretrain to use pre-trained model
python main.py \
    --loader 2 \
    --loader_config_path configs/loader2.yaml \
    --module 4 \
    --module_config_path configs/module4.yaml \
    --name 2_4 \
    --vgg_net_arch 11
```

```shell
# loader2 & module5: CIFAR-10 & ResNet
# You can add --use_pretrain to use pre-trained model
python main.py \
    --loader 2 \
    --loader_config_path configs/loader2.yaml \
    --module 5 \
    --module_config_path configs/module5.yaml \
    --name 2_5 \
    --res_net_arch 18
```
