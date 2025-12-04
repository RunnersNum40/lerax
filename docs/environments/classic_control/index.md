---
description: Classic Control Environments in the Lerax Rienforcement Learning Library
---

# Classic Control

Lerax versions of the [Gymnasium Classic Control Environments](https://gymnasium.farama.org/environments/classic_control/).
In Lerax the dynamics are integrated using [Diffrax](https://docs.kidger.site/diffrax/) for more physically accurate results.
During enviromnent creation, you can specify the `dt: float`, `solver: diffrax.AbstractSolver` parameter, and `stepsize_controller: diffrax.AbstractStepSizeController` parameters to control the integration.
To use the exact Gymnasium dynamics, leave the default `dt` and specify [`diffrax.Euler`](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Euler) for the solver.
