# PTNCN-Local-recurrent-and-parallel-predictive-coding
We introduce a Local recurrent Predictive coding model termed as Parallel temporal Neural Coding Network. Unlike classical RNNs, our model is pure local and doesn't require computing gradients backward in time; thus, it is computationally more efficient compared to BPTT and can be used for online learning.

# Requirements
Our implementation is easy to follow and, with knowledge of basic linear algebra, one can decode the inner workings of the PTNCN algorithm. In this framework, we have provided simple modules; thus hopefully making it very convenient to extend our framework to layers>3.
To run the code, you should only need following basic packages:
1. TensorFlow (version >= 2.0)
2. Numpy
3. Matplotlib
4. Python (version >=3.5)

# Training the system on Penn Tree Bank
Simply run python ContPTNCN/src/train_discrete_ptncn.py
Tips while using this algorithm/model on your own datasets:
1. Track your local losses, and accordingly adjust the hyper-parameters for the model.
2. Play with non-zero, small values for the weight decay coefficients.
3. Play with initialization values for backward/error weights (E)
4. Increasing inference steps (K) that optimize free-energy.


# Citation

If you use or adapt (portions of) this code/algorithm in any form in your project(s), or
find the PTNCN/Parallel Temporal Predictive Coding algorithm helpful in your own work, please cite this code's source paper:
```bibtex
@article{ororbia2020continual,
  title={Continual learning of recurrent neural networks by locally aligning distributed representations},
  author={Ororbia, Alexander and Mali, Ankur and Giles, C Lee and Kifer, Daniel},
  journal={IEEE transactions on neural networks and learning systems},
  volume={31},
  number={10},
  pages={4267--4278},
  year={2020},
  publisher={IEEE}
}
```
