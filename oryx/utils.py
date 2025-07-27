import equinox as eqx
import jax


def clone_state(state: eqx.nn.State) -> eqx.nn.State:
    """Clone the state."""
    leaves, treedef = jax.tree.flatten(state)
    state_clone = jax.tree.unflatten(treedef, leaves)
    return state_clone
