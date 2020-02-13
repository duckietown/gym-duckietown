# Imitation Learning

## Introduction
In this baseline we train a small squeezenet model on expert trajectories to simply clone the behavior of the expert.
Using only the expert trajectories would result in a model unable to recover from non-optimal positions,Hence we use a technique called DAgger a dataset aggregation technique with mixed policies between expert and model.
This technique of random mixing would help the model learn a more general trajectory than the optimal one provided by the expert alone.

## Quick start
1) Clone this [repo](https://github.com/duckietown/gym-duckietown):

$ git clone https://github.com/duckietown/gym-duckietown.git

2) Change into the directory:

$ cd gym-duckietown

3) Install the package:

$ pip3 install -e .

4) Start training:

$ python -m learning.imitation.iil-dagger.train

5) Test the trained agent specifying the saved model:

$ python -m learning.imitation.pytorch-v2.test --model-path ![path]

## Acknowledgment
- We started from previous work done by Manfred Díaz as a boilerplate, and we would like to thank him for his full support with code and answering our questions

## Authors
- [Mostafa ElAraby ](https://www.mostafaelaraby.com/)
- [Linkedin](https://linkedin.com/in/mostafaelaraby)
- Ramon Emiliani
- [Linkedin](https://www.linkedin.com/in/ramonemiliani)

## References
@phdthesis{diaz2018interactive,
  title={Interactive and Uncertainty-aware Imitation Learning: Theory and Applications},
  author={Diaz Cabrera, Manfred Ramon},
  year={2018},
  school={Concordia University}
}
