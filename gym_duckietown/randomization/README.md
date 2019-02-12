# Randomization API

This document describes the API for domain randomization, as well as what is currently implemented.

## API

The domain randomization in `gym-duckietown` is driven by the [Randomizer](https://github.com/duckietown/gym-duckietown/blob/domain-randomization-api/gym_duckietown/randomization/randomizer.py#L8) class, which takes as input a config file, and outputs (upon a call to [`randomize()`](https://github.com/duckietown/gym-duckietown/blob/domain-randomization-api/gym_duckietown/randomization/randomizer.py#L22)) settings which are used by the [`Simulator`](https://github.com/duckietown/gym-duckietown/blob/domain-randomization-api/gym_duckietown/simulator.py) class when generating training environments. You still need to pass [`domain_rand=True`](https://github.com/duckietown/gym-duckietown/blob/domain-randomization-api/gym_duckietown/simulator.py#L129) when you call the `Simulator`'s [constructor](https://github.com/duckietown/gym-duckietown/blob/domain-randomization-api/gym_duckietown/simulator.py#L129).

The API is simple - it looks through the config file, and randomizes according to the values set in the config. Below, you can find our supported protocol, as well as what is currently randomized in `gym-duckietown`. 

## Randomization Protocol

The protocol reads 

To implement your own variant of domain randomization, you should follow these steps:

1. 

Alternatively, feel free to open an issue (or a pull request!) and we can help you get what you need into `gym-duckietown`.

## Randomization Details
* Camera Noise
    * Qualitative Description
    * Variable Name
    * Default Value
    * Default Randomization Range
    * Code Reference

* Global Light Position
    * Qualitative Description
    * Variable Name
    * Default Value
    * Default Randomization Range
    * Code Reference
    
* Horizon Mode
    * Qualitative Description
    * Variable Name
    * Default Value
    * Default Randomization Range
    * Code Reference
