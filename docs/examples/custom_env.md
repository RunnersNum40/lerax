---
title: Making a Custom Environment
description: Example of creating a custom environment in Lerax reinforcement learning library.
status: draft
---

We will create an version of the [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment with Lerax.

## State

Lerax environments cannot have internal state.
Instead the state must be a [PyTree](https://jax.readthedocs.io/en/latest/pytrees.html) that is passed around by the environment methods. We can create a state by subclassing `lerax.env.AbstractEnvState`.

```py
from jax import numpy as jnp

from lerax.env import AbstractEnvState

class CartPoleState(AbstractEnvState):
    position: jnp.ndarray
    velocity: jnp.ndarray
    angle: jnp.ndarray
    angular_velocity: jnp.ndarray
```
