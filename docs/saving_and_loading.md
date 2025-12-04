---
title: Saving and Loading Policies
description: How to serialize and deserialize Lerax policies and other modules.
---

# Saving and Loading Policies

Lerax uses [Equinox](https://github.com/patrick-kidger/equinox) to store models as PyTrees on disk.
Any class that inherits from [`lerax.utils.Serializable`](api/utils.md#lerax.utils.Serializable) gains:

- `serialize(path)` — save the object to a `.eqx` file.
- `deserialize(path, ...)` (class method) — load the object back from disk.

All policies (`AbstractStatelessPolicy` / `AbstractStatefulPolicy` subclasses) are `Serializable`.

### Saving a Model

```py
from jax import random as jr

from lerax.env import CartPole
from lerax.policy import MLPActorCriticPolicy

policy_key = jr.key(0)

env = CartPole()
policy = MLPActorCriticPolicy(env=env, key=policy_key)

policy.serialize("model.eqx")
```

## Loading a Model

Use the `deserialize` class method on the policy (or other `Serializable` subclass):

```py
from jax import random as jr

from lerax.env import CartPole
from lerax.policy import MLPActorCriticPolicy

env = CartPole()

loaded_policy = MLPActorCriticPolicy.deserialize("model.eqx", env, key=jr.key(1)) # (1)!
```

1. Constructor arguments must match the original class signature

After the path argument, you must provide usual constructor arguments for the class.
Values that determine PyTree structure must match the original object (e.g. the `env` argument above), but other arguments (e.g. random keys) can differ and will be overwritten by the loaded parameters.

## Serializable Mixin

If you write your own modules and want the same save/load API, inherit from [`Serializable`](api/utils.md#lerax.utils.Serializable):

```py
import equinox as eqx
from lerax.utils import Serializable

class MyModule(Serializable):
    a: float
    b: float

    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

m = MyModule(1.0, 2.0)
m.serialize("my_module.eqx")

m = MyModule.deserialize("my_module.eqx", 0.0, 0.0)
```

This works for policies, algorithms, or any other Equinox module that you want to persist.
