"""Checkpointable private RNG streams for optional training updates."""
import numpy as np


def restore_rng(state, seed):
    """Resume PCG64 exactly; legacy checkpoints start a seeded stream.

    Invalid saved state raises rather than silently replacing training history.
    This covers the private stream only, not Torch RNG or the input cursor.
    """
    rng = np.random.default_rng(seed)
    if state is not None:
        rng.bit_generator.state = state
    return rng
