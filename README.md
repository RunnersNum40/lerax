# Oryx

This is a work in progress implementation of a JAX based reinforcement learning library using Equinox.
The main feature is Neural Differential Equation based models.
NDEs can be extraordinarily computationally intensive, this library is intended to provide an optimised implementation of NDEs and other RL algorithms using just in time compilation (JIT) that can be fused with environments that support it to achieve high performance using the Anakin model for fully GPU based RL.

I'm working on this in my free time, so it may take a while to get to a usable state. I'm also mainly developing this for personal research, so it may not be suitable for all use cases.

## Code Style

This code is written to follow the [Equinox's abstract/final pattern](https://docs.kidger.site/equinox/pattern/) for code structure and [Black formatting](https://black.readthedocs.io/en/stable/index.html#).
This is intended to make the code more readable and maintainable, and to ensure that it is consistent with the Equinox library.
If you want to contribute, please follow these conventions.

## Credit

A ton of the code is a slight translation of the code found in the [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) and [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) libraries which are both under the MIT license.
The developers of these excellent libraries have done a great job of creating a solid foundation for reinforcement learning in Python, and I have learned a lot from their code.

In addition, the NDE code is heavily inspired by the work of [Patrick Kidger](https://kidger.site/publications/) and the entire library is based on his excellent [Equinox library](https://github.com/patrick-kidger/equinox) along with some use of [Diffrax](https://github.com/patrick-kidger/diffrax) and [jaxtyping](https://github.com/patrick-kidger/jaxtyping).

## TODO

- Get it working!
  - Environment implementations
    - Custom environments
    - Wrappers for other libraries
      - Gymnax
      - Gymnasium
  - Wrappers
  - Policies
  - Algorithms
    - PPO
- Logging
  - Code flow logging
  - Training logging
- Documentation
  - Standardize docstring formats
  - Write documentation for all public APIs
  - Publish docs
  - Examples
- Testing
  - More thorough unit testing
  - Integration testing
  - Runtime jaxtyping
- Use it
  - Personal research
- Optimise for performance under JIT compilation
  - Good vectorization support
  - Sharding support for distributed training
- Round out features
  - Rendering support
  - Expand RL variants to include more algorithms
  - Create a more comprehensive set of environments
- Migrate from tensorboard to aim
