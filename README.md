# Learning Human Poses from Actions

This repository is the PyTorch implementation for the network presented in:

> Aditya Arun, C.V. Jawahar, and M. Pawan Kumar,          
> **Learning Human Poses from Actions**,        
> [BMVC 2018](arXiv:1807.09075, 2018).

## Requirements
- cudnn
- [PyTorch](http://pytorch.org/)
- Python with h5py, opencv and [progress](https://anaconda.org/conda-forge/progress)
- Optional: [tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) 

## Training
- Modify `src/ref.py` to setup the MPII dataset path and root directory. 
- Run `Prediction Net`
```
python main.py -arch pred_net
```
-Run `Conditional Net`
```
python main.py -arch cond_net
```
