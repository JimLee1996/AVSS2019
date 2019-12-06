## VioNet

Efficient Violence Detection Using 3D Convolutional Neural Networks.

### Model Implemented

- C3D

- ConvLSTM

- 3D DenseNet (Lean & Original)

### Code Structure

- log: saving train and val logs

- models: models implemented

- pth: saving checkpoints

- config.py: model, dataset selection etc.

- dataset.py: handle data loading, please refer to [pytorch docs](https://pytorch.org/docs/stable/data.html)

- epoch.py: phases of traing and validation

- main.py: run from here

- model.py: wrapper of dir models

- spatial_transforms.py: spatiial data augumentation

- target_transforms.py: handle labels

- temporal_transforms.py: temporal data augumentation

- utils.py: common tools, like logging

- *.pth: init weights pre-trained on Kinetics

- torchsummary.py: like `summary` model in keras

### How to run

- Download pretrained weights on Kinetices and put them into dir `weights`. [[Weights](https://drive.google.com/file/d/1pNrAzWHQJLzOEH_-407rel3VV45YuJ6f/view?usp=sharing)]

- Download dataset and put them into dir `../VioDB`. [[Dataset](https://drive.google.com/file/d/1zxttpCDegQUmPHE8WwD8EMPpCngRRVkm/view?usp=sharing)]

- Before runing `main.py`, you might need to modify the parameters at the end of the code.
