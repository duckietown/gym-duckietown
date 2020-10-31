# Randomization API

This document describes the API for domain randomization, as well as what is currently implemented.

## API

The domain randomization in `gym-duckietown` is driven by the [`Randomizer`](https://github.com/duckietown/gym-duckietown/tree/master/gym_duckietown/randomization/randomizer.py#L8) class, which takes as input a config file, and outputs (upon a call to [`randomize()`](https://github.com/duckietown/gym-duckietown/tree/master/gym_duckietown/randomization/randomizer.py#L22)) settings which are used by the [`Simulator`](https://github.com/duckietown/gym-duckietown/tree/master/gym_duckietown/simulator.py) class when generating training environments. You still need to pass [`domain_rand=True`](https://github.com/duckietown/gym-duckietown/tree/master/gym_duckietown/simulator.py#L129) when you call the `Simulator`'s [constructor](https://github.com/duckietown/gym-duckietown/tree/master/gym_duckietown/simulator.py#L129).

The API is simple - it looks through the config file, and randomizes according to the values set in the config. Below, you can find our supported protocol, as well as what is currently randomized in `gym-duckietown`.

## Randomization Protocol

The protocol reads the config files provided in the constructor for the [`Randomizer`](https://github.com/duckietown/gym-duckietown/tree/master/gym_duckietown/randomization/randomizer.py#L8), and sets the corresponding randomization ranges. If a variable that _can be randomized_ is **not** found in the file passed in as `randomization_config_fp`, it will use the default randomization values, which can be found [here](https://github.com/duckietown/gym-duckietown/tree/master/gym_duckietown/randomization/config/default.json). It is highly recommended to **not edit the defaults**.

To implement your own variant of domain randomization, you should follow these steps:

1. Make a copy of the [`default_dr.json`](https://github.com/duckietown/gym-duckietown/tree/master`/gym_duckietown/randomization/config/default_dr.json), edit it to your liking.
2. Place your new file inside the [`randomization/config`](https://github.com/duckietown/gym-duckietown/tree/`/gym_duckietown/randomization/config) directory, and pass it's filename (`.json` included!) to the `Randomizer` [constructor call](https://github.com/duckietown/gym-duckietown/tree/master/gym_duckietown/simulator.py#L186) under `randomization_config_fp`.

We currently support three types of distributions for randomization: `int`, `uniform`, and `normal`, which correspond to the functions within `numpy.random`. Please note that all fields except `size` (defaults to 1) need to provided, and that `normal` needs `loc` and `scale` rather than `high` and `low`.

Alternatively, feel free to open an issue (or a pull request!) and we can help you get what you need into `gym-duckietown`.

## Randomization Details
* Camera Noise
    * Qualitative Description - Adds noise to the camera position for data augmentation
    * Variable Name - `camera_noise`
    * Distribution Type - `uniform`
    * Default Value - `[0, 0, 0]`
    * Default Randomization Range - `[-0.005,0.005]` for each of the three degrees of freedom
    * [Code Reference](https://github.com/duckietown/gym-duckietown/tree/master/gym_duckietown/simulator.py#L1266)

* Global Light Position
    * Qualitative Description - Where the global light sits in 3D space. Makes the simulator more or less illuminated
    * Variable Name - `light_pos`
    * Distribution Type - `uniform`
    * Default Value - `[-40, 200, 100]`
    * Default Randomization Range - `[-150, 170, -150]` to `[150, 220, 150]`
    * [Code Reference](https://github.com/duckietown/gym-duckietown/tree/master/gym_duckietown/simulator.py#L340)

* Horizon Mode
    * Qualitative Description - Horizon color; makes the task more or less difficult by making the horizon look more or less like the road.
    * Variable Name - `horz_mode`
    * Distribution Type - `randint(0, 4)`
    * Default Value - `0`
    * Default Randomization Range - `[0, 4)`
    * [Code Reference](https://github.com/duckietown/gym-duckietown/tree/master/gym_duckietown/simulator.py#L326)
