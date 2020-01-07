## ViO

### [model.py](model.py)

VioNet model

### [backend.py](backend.py)

the process from input to output

`input`: (batch_size)x3x16x112x112 tensor

`output`: class

### [videoloader.py](videoloader.py)

Based on OpenCV

`input`: video path

`output`: clips with 16 frames

### [demo.py](demo.py)

how to run the code

```
python demo.py path_to_video
```
