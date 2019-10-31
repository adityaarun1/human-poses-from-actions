# Learning Human Poses from Actions

This repository is the PyTorch implementation for the network presented in:

> Aditya Arun, C.V. Jawahar, and M. Pawan Kumar,          
> **Learning Human Poses from Actions**,        
> [BMVC 2018](arXiv:1807.09075, 2018).

\[[Project Page](https://cvit.iiit.ac.in/research/projects/cvit-projects/pose-from-action)\] \[[Paper](http://bmvc2018.org/contents/papers/0898.pdf)\] \[[Supplementary](http://bmvc2018.org/contents/supplementary/pdf/0898_supp.pdf)\] \[[ArXiv](https://arxiv.org/abs/1807.09075)\]

If you use this code, please consider citing our work.
```
@inproceedings{posesfromactions2018,
  title={Learning Human Poses from Actions},
  author={Arun, Aditya and Jawahar, C.~V and Kumar, M. Pawan},
  booktitle={BMVC},
  year={2018}
}
```

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
- Run `Conditional Net`
```
python main.py -arch cond_net
```
