import jax
import jax.random as jr
import blackjax
import jax.numpy as jnp   # For operations on *traced* arrays
from blackjax.smc import resampling
import flax.nnx as nnx
from particles import Particles

def _smc_inference_loop(rng_key, smc_kernel, initial_state):
    """Run the tempered SMC algorithm.

    We run the adaptive algorithm until the tempering parameter lambda reaches the value
    lambda=1.

    """
    def cond(carry):
        i, state, _k = carry
        return state.lmbda < 1

    def one_step(carry):
        i, state, k = carry
        k, subk = jax.random.split(k, 2)
        state, _ = smc_kernel(subk, state)
        return i + 1, state, k

    n_iter, final_state, _ = jax.lax.while_loop(
        cond, one_step, (0, initial_state, rng_key)
    )

    return n_iter, final_state

def smc(particles, data, graphdef, mesh=None, key=jr.PRNGKey(42)):
    part_object = Particles(particles, data, graphdef)
    
    inv_mass_dim = sum([p[0].size for p in jax.tree_util.tree_flatten(particles)[0]])
    hmc_parameters = dict(step_size=1e-2, inverse_mass_matrix=jnp.eye(inv_mass_dim), num_integration_steps=10
        )
    
    rng_key, init_key, sample_key = jax.random.split(key, 3)
    
    def hmc_step(key, state, logdensity):
        hmc = blackjax.hmc(logdensity, **hmc_parameters)
        step = nnx.jit(hmc.step)
        return step(key, state)

    tempered = blackjax.adaptive_tempered_smc(
        part_object.log_prior,
        part_object.log_likelihood,
        hmc_step,
        blackjax.hmc.init,
        mcmc_parameters={},
        resampling_fn=resampling.systematic,
        target_ess=0.9,
        num_mcmc_steps=10,
    )
    
    
    initial_smc_state = particles
    initial_smc_state = tempered.init(initial_smc_state)
    n_iter, final_state = _smc_inference_loop(sample_key, tempered.step, initial_smc_state)
    
    print(final_state, "\n************************")
    print("Number of steps in the adaptive algorithm: ", n_iter.item())
    return final_state