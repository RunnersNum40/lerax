# Lerax

This is a work in progress implementation of a JAX based reinforcement learning library using Equinox.
The main feature is Neural Differential Equation based models.
This is meant as a cleaner and more complete continuation of earlier work in this repo [NCDE-RL](https://github.com/RunnersNum40/NCDE-RL)
NDEs can be extraordinarily computationally intensive, this library is intended to provide an optimized implementation of NDEs and other RL algorithms using just in time compilation (JIT).
Paired with environments that support JIT, high performance is possible using the Anakin architecture for fully GPU based RL.

I'm working on this in my free time, so it may take a while to get to a usable state. I'm also mainly developing this for personal research, so it may not be suitable for all use cases.

## Credit

A ton of the code is a slight translation of the code found in the [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) and [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) libraries which are both under the MIT license.
The developers of these excellent libraries have done a great job of creating a solid foundation for reinforcement learning in Python, and I have learned a lot from their code.

In addition, the NDE code is heavily inspired by the work of [Patrick Kidger](https://kidger.site/publications/) and the entire library is based on his excellent [Equinox library](https://github.com/patrick-kidger/equinox) along with some use of [Diffrax](https://github.com/patrick-kidger/diffrax) and [jaxtyping](https://github.com/patrick-kidger/jaxtyping).

## Usage

### Installation

Install via pip:

```bash
pip install lerax@git+https://github.com/RunnersNum40/lerax.git
```

Or clone the repo and install in editable mode:

```bash
git clone https://github.com/RunnersNum40/lerax.git
```
```bash
cd lerax
```
```bash
pip install -e .
```

### Running an example

```bash
python examples/ppo.py
```

### Running TensorBoard

```bash
tensorboard --logdir runs
```

Then open your browser to `http://localhost:6006`.

### Creating your own models and environments

Check out the [MLP Actor Critic](lerax/policy/actor_critic/mlp.py) for a simple example of how to create your own actor critic model.
Check out the [PPO example](examples/ppo.py) for how to use your model in training.
Check out the [CartPole environment](lerax/env/cartpole.py) for how to create your own environment.
Check out the [Gymnasium wrapper](lerax/compatibility/gym.py) for how to wrap Gymnasium environments (this will be slower to run than a fully Jax environment).

## TODO

- Optimise for performance under JIT compilation
  - Sharding support for distributed training
- Expand policy support beyond Box and Discrete spaces
- Documentation
  - Standardize docstring formats
  - Write documentation for all public APIs
  - Add API to docs when Zensical supports it
- Testing
  - Unit testing
  - Integration testing
  - Full Jaxtyping
    - Ensure all functions and classes have proper type annotations
- Use it
  - Personal research
- Round out features
  - Expand RL variants to include more algorithms
  - Create a more comprehensive set of environments
    - Brax based environments
  - Save and load models

## Code Style

This code is written to follow the [Equinox's abstract/final pattern](https://docs.kidger.site/equinox/pattern/) for code structure and [Black formatting](https://black.readthedocs.io/en/stable/index.html#).
This is intended to make the code more readable and maintainable, and to ensure that it is consistent with the Equinox library.
If you want to contribute, please follow these conventions.
