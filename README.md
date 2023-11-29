# PTNCN-Local-recurrent-and-parallel-predictive-coding
We introduce a locally recurrent predictive coding model, a neuro-mimetic system that was specifically named the <i>Parallel Temporal Neural Coding Network</i> (P-TNCN). Unlike classical recurrent neural networks (RNNs), our proposed model conducts inference over its neuronal dynamics and adapts its synpatic connection strengths in local manner, i.e., it utilizes Hebbian-like synaptic adjustments. As a result, the underlying temporal model does not require computing gradients backward in time (also known as "unfolding" or "unrolling" back through time); thus, the model learns in a more computationally more efficient way than RNNs trained with BPTT and can be used for online learning.

# Requirements
Our implementation is easy to follow and, with knowledge of basic linear algebra, one can decode the inner workings of the PTNCN algorithm. In this framework, we have provided simple modules; thus hopefully making it very convenient to extend our framework to layers>3.
To run the code, you should only need following basic packages:
1. TensorFlow (version >= 2.0)
2. Numpy
3. Matplotlib
4. Python (version >=3.5)

# Training the system on Penn Tree Bank (PTB)
Simply run python ContPTNCN/src/train_discrete_ptncn.py

Tips while using this algorithm/model on your own datasets:
1. Track your local losses, and accordingly adjust the hyper-parameters for the model.
2. Play with non-zero, small values for the weight decay coefficients.
3. Play with initialization values for backward/error synaptic weights (variables that contain capital "E" in their name)
4. Increasing the number of inference steps (K), which affect the optimization of the model's underlying free-energy functional.


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
