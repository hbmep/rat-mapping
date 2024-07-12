import jax

PLATFORM = "cpu"
jax.config.update("jax_platforms", PLATFORM)
