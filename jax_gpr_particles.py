
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
from jax import config
#config.update("jax_enable_x64", True)
import jax.tree_util as jtu
from abc import ABC
from flax import nnx
from flax import linen
import jax.random as jr
from jaxtyping import install_import_hook
with install_import_hook("gpjax", "beartype.beartype"):
    import GPJax.gpjax as gpx
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import jax.numpy as jnp
from datetime import datetime
import blackjax 
from blackjax.smc import resampling
import tensorflow_probability.substrates.jax.distributions as tfd
import numpy as np
import seaborn as sns
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
import matplotlib.pyplot as plt
import matplotlib
from typing import Callable, NamedTuple, Optional, List
from jax_changepoint import ChangePoints

@nnx.jit
def create_sharded_model(particles):
  state = nnx.state(particles)                   # The model's state, a pure pytree.
  pspecs = nnx.get_partition_spec(state)     # Strip out the annotations from state.
  sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
  nnx.update(particles, sharded_state)           # The model is sharded now!
  return particles

def set_priors(posterior, prior_dict):
    graphdef, params, *static_state = nnx.split(posterior, nnx.VariableState, ...)
    params = gpx.parameters.set_priors(params, param_priors=prior_dict)
    posterior_new = nnx.merge(graphdef, params)
    return posterior_new

def init_particles(posterior, num_particles, key) -> ArrayLikeTree:
    graphdef, params, *static_state = nnx.split(posterior, nnx.VariableState, ...)
    params_flat, struct = jtu.tree_flatten(params, lambda x: isinstance(x, nnx.VariableState))
    particles = []
    for param in params_flat:   # Iterate param1, param2, ... paramN
        key, subkey = jr.split(key)
        if param._tag == "static":
            particles.append(param)
            continue
        else:
            param.value = param.prior.sample(seed=subkey, sample_shape=tuple([num_particles] + list(param.value.shape)))
            particles.append(param)
    particle_params = jtu.tree_unflatten(struct, particles)
    particles = nnx.merge(graphdef, particle_params)
    return particles   # list of posteriors


class Particles: ## Container object for particles
    def __init__(self, particles: nnx.State, data: gpx.Dataset, graphdef):
        self.particles = particles
        self.data = data
        self.graphdef = graphdef
        
    def log_prior(self, particles):
        params = gpx.parameters.transform(particles, gpx.parameters.DEFAULT_BIJECTION)
        particles = nnx.merge(self.graphdef, params)   # assume you know the
        return jnp.sum(particles.log_prior())
    
    def log_likelihood(self, particles):
        params = gpx.parameters.transform(particles, gpx.parameters.DEFAULT_BIJECTION)
        particles = nnx.merge(self.graphdef, params)   # assume you know the
        if isinstance(particles, gpx.gps.ConjugatePosterior):
            return gpx.objectives.conjugate_mll(particles, self.data)
        else:
            return gpx.objectives.log_posterior_density(particles, self.data)
    
    def tree_flatten(self):
        return (self.particles), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children)

    
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

def smc(particles, data, graphdef, mesh, key):
    part_object = Particles(particles, data, graphdef)
    hmc_parameters = dict(step_size=1e-2, inverse_mass_matrix=jnp.eye(8), num_integration_steps=10
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

def plot_posterior(p, ax, D):
    
    cols = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    xtest = jnp.linspace(0, 100, 100).reshape(-1, 1)
    latent_dist = p.predict(xtest, train_data=D)
    predictive_dist = p.likelihood(latent_dist)

    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()
    ax.plot(D.X, D.y, "x", label="Observations")
    ax.fill_between(
        xtest.squeeze(),
        predictive_mean - 2 * predictive_std,
        predictive_mean + 2 * predictive_std,
        alpha=0.1,
        label="Two sigma",
        color=cols[1],
    )
    ax.plot(
        xtest,
        predictive_mean - 2 * predictive_std,
        linestyle="--",
        linewidth=1,
        color=cols[1],
    )
    ax.plot(
        xtest,
        predictive_mean + 2 * predictive_std,
        linestyle="--",
        linewidth=1,
        color=cols[1],
    )
    # ax.plot(
    #     xtest, ytest, label="Latent function", color=cols[0], linestyle="--", linewidth=2
    # )
    ax.plot(xtest, predictive_mean, label="Predictive mean", color=cols[1])
    
def untangle(state, N, D):
    separate = [jtu.tree_map(lambda lst: lst[i], state.particles) for i in range(N)]
    posteriors = [nnx.merge(graphdef, s) for s in separate]

    fig, ax = plt.subplots(figsize=(7.5, 2.5))
    argsort = jnp.argsort(state.weights)
    for p in posteriors[:3]:  # Plot most likely particles
        plot_posterior(p, ax, D)
    plt.show()
    return
    


def plot_distributions(state: gpx.gps.AbstractPosterior):
    graphdef, params, *static_state = nnx.split(state, gpx.parameters.Parameter, ...)
    keyleaves, struct = jtu.tree_flatten_with_path(params, lambda x: isinstance(x, nnx.VariableState))
    fig, ax = plt.subplots()
    for i, (key, param) in enumerate(keyleaves):   # Iterate param1, param2, ... paramN
        if str(key[-1].key) == "locations":
            ax.plot(D.X, D.y)
            ax.hist(param.value, bins=100)
            ax.set_xlim(0, 100)
            ax.set_title(key[-1].key)
    plt.show()
    return 
    
if __name__ == "__main__":
    key = jr.PRNGKey(42) 
    t = datetime.now()
    print(t)
    N_particles = 8
    key, subkey = jr.split(key)
    data = np.load("/home/janneke/changepoint-gp/notebooks/multiple-cp/data.npz")
    X = data["X"]
    y = data["y"]
    # plt.plot(X, y, "kx")
    # plt.show()
    D = gpx.Dataset(X=X, y=y)

    #xtest = jnp.linspace(-3.5, 3.5, 500).reshape(-1, 1)

    kernel = gpx.kernels.RBF() 
    #kernel = gpx.kernels.RBF()
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)

    posterior = prior * likelihood
    graphdef, params, *static_state = nnx.split(posterior, gpx.parameters.Parameter, ...)

    prior_dict = {"prior": {"kernel": {"variance": tfd.LogNormal(1, 1), "lengthscale": tfd.LogNormal(1, 1)},
                            "mean_function": {"constant": tfd.Normal(0, 1)}},
                    "latent": {"obs_stddev": tfd.Normal(1, 1)}}
                    

    # prior_dict = {"prior": {"kernel": {"kernels": {0: {"variance": tfd.LogNormal(0, 1), "lengthscale": tfd.LogNormal(1, 1)},
    #                                                1: {"variance": tfd.LogNormal(0, 1), "lengthscale": tfd.LogNormal(1, 1)}},
    #                                     "locations": tfd.Uniform(0, len(X)),
    #                                     "steepness": tfd.Uniform(0, 10)},
    #                         "mean_function": {"constant": tfd.Normal(0, 1)}},
    #                         "likelihood": {"obs_stddev": tfd.LogNormal(1, 1)}}

    posterior = set_priors(posterior, prior_dict)
    particle_list = init_particles(posterior, N_particles, key) 
    with mesh:
        state_sharded = create_sharded_model(particle_list)
    params_bijection = gpx.parameters.DEFAULT_BIJECTION
    graphdef, params, *static_state = nnx.split(particle_list, nnx.VariableState, ...)
    params = gpx.parameters.transform(params, params_bijection, inverse=True)
    key, smc_key = jr.split(key)
    final_state = smc(params, D, graphdef, smc_key)
    params = gpx.parameters.transform(params, params_bijection)
    final_particles = nnx.merge(graphdef, final_state.particles)  
    dt = datetime.now() - t
    print("Sampling took", dt.total_seconds())