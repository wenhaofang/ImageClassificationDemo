## Image Classification Demo

This repository includes some demo image classification models.

Note: The project refers to https://github.com/bentrevett/pytorch-image-classification

<br/>

Datasets

* `dataset1`: Mnist

Models

* `model1`: Multilayer Perceptron

### Data Process

```shell
PYTHONPATH=. python dataprocess/process.py
```

### Unit Test

* for loader

```shell
# loader1
PYTHONPATH=. python loaders/loader1.py
```

* for module

```shell
# module1
PYTHONPATH=. python modules/module1.py
```

### Main Process

```shell
python main.py
```

You can change the config either in the command line or in the file `utils/parser.py`
