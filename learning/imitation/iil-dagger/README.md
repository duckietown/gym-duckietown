# Imitation Learning

## Introduction

In this baseline we train a small squeezenet model on expert trajectories to simply clone the behavior of the expert.
Using only the expert trajectories would result in a model unable to recover from non-optimal positions; Instead, we use a technique called DAgger: a dataset aggregation technique with mixed policies between expert and model.

## Quick start

Use the jupyter notebook notebook.ipynb to quickly start training and testing the imitation learning Dagger.

## Detailed Steps

### Clone the repo

Clone this [repo](https://github.com/duckietown/gym-duckietown):

$ git clone https://github.com/duckietown/gym-duckietown.git

$ cd gym-duckietown

### Installing Packages

$ pip3 install -e .

## Training

$ python -m learning.imitation.iil-dagger.train

### Arguments

* --episode: number of episodes
* --horizon: number of steps per episode
* --learning-rate: index of learning rate from list [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
* --decay: mixing decay between expert and learner [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
* --save-path: directory used to save output model
* --map-name: name of the map used during the training
* --num-outputs: specify number of outputs from the learner model 1 to predict only angular velocity with fixed speed and 2 to predict both of them
* --domain-rand: flag to enable domain randomization to rbe able to transfer trained model to real world.
* --randomize-map: randomize training maps on reset

## Testing

$ python -m learning.imitation.iil-dagger.test

### Arguments

*  --model-path: path of the model to be tested
* --episode: number of episodes
* --horizon: number of steps per episode

## Submitting 
Use [Pytorch RL Template](https://github.com/duckietown/challenge-aido_LF-template-pytorch) and replace model with the model trained in model/squeezenet.py
and use the following code snippet to convert speed and angular velocity to pwm left and right.
``` Python
velocity, omega = self.compute_action(self.current_image) 

# assuming same motor constants k for both motors
k_r = 27.0
k_l = 27.0
gain = 1.0
trim = 0.0

# adjusting k by gain and trim
k_r_inv = (gain + trim) / k_r
k_l_inv = (gain - trim) / k_l
wheel_dist = 0.102
radius=0.0318

omega_r = (velocity + 0.5 * omega * wheel_dist) / radius
omega_l = (velocity - 0.5 * omega * wheel_dist) / radius

# conversion from motor rotation rate to duty cycle
u_r = omega_r * k_r_inv
u_l = omega_l * k_l_inv

# limiting output to limit, which is 1.0 for the duckiebot
pwm_right = max(min(u_r, 1), -1)
pwm_left = max(min(u_l, 1), -1)

```

## Acknowledgment

* We started from previous work done by Manfred Díaz as a boilerplate, and we would like to thank him for his full support with code and answering our questions.

## Authors

* [Mostafa ElAraby ](https://www.mostafaelaraby.com/)
  + [Linkedin](https://linkedin.com/in/mostafaelaraby)
* Ramon Emiliani
  + [Linkedin](https://www.linkedin.com/in/ramonemiliani)

## References

``` 

@phdthesis{diaz2018interactive,
  title={Interactive and Uncertainty-aware Imitation Learning: Theory and Applications},
  author={Diaz Cabrera, Manfred Ramon},
  year={2018},
  school={Concordia University}
}

@inproceedings{ross2011reduction,
  title={A reduction of imitation learning and structured prediction to no-regret online learning},
  author={Ross, St{\'e}phane and Gordon, Geoffrey and Bagnell, Drew},
  booktitle={Proceedings of the fourteenth international conference on artificial intelligence and statistics},
  pages={627--635},
  year={2011}
}

@article{iandola2016squeezenet,
  title={SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size},
  author={Iandola, Forrest N and Han, Song and Moskewicz, Matthew W and Ashraf, Khalid and Dally, William J and Keutzer, Kurt},
  journal={arXiv preprint arXiv:1602.07360},
  year={2016}
}
```
