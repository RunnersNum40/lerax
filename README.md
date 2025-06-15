# Oryx

This is a work in progress implementation of a JAX based reinforcement learning library using Equinox.
The main feature is Neural Differential Equation based models.
NDEs can be extraordinarily computationally intensive, this library is intended to provide an optimised implementation of NDEs and other RL algorithms using just in time compilation (JIT) that can be fused with environments that support it to achieve high performance using the Anakin model for fully GPU based RL.

I'm working on this in my free time, so it may take a while to get to a usable state. I'm also mainly developing this for personal research, so it may not be suitable for all use cases.
