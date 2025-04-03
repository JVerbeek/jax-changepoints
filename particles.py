import jax
# from jax import config
# config.update("jax_enable_x64", True)  We should turn this on again at some point, but now it led to some errors.
import jax.tree_util as jtu
from flax import nnx
import jax.random as jr
from jaxtyping import install_import_hook
with install_import_hook("gpjax", "beartype.beartype"):  # Local import!
    import GPJax.gpjax as gpx
import jax.numpy as jnp
from datetime import datetime
import blackjax 
from blackjax.smc import resampling
import tensorflow_probability.substrates.jax.distributions as tfd
import numpy as np
from blackjax.types import ArrayLikeTree
import matplotlib.pyplot as plt
import matplotlib

@nnx.jit
def create_sharded_model(particles):
  state = nnx.state(particles)                   # The model's state, a pure pytree.
  pspecs = nnx.get_partition_spec(state)     # Strip out the annotations from state.
  sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
  nnx.update(particles, sharded_state)           # The model is sharded now!
  return particles

def set_priors(posterior: gpx.gps.AbstractPosterior, prior_dict):
    """Set priors for variables in GPJax Posterior.

    Arguments:
        posterior -- GPJax posterior, should be instance of Abstract posterior.
        prior_dict -- Nested dictionary, containing variable: tfd.Distribution pairs.  

    Returns:
        Posterior with priors set.
    """
    graphdef, params, *static_state = nnx.split(posterior, nnx.VariableState, ...)
    params = gpx.parameters.set_priors(params, param_priors=prior_dict)
    posterior_new = nnx.merge(graphdef, params)
    return posterior_new

def init_particles(posterior: gpx.gps.AbstractPosterior, num_particles: int, rng_key: jax.random.PRNGKey) -> ArrayLikeTree:
    """Initialize particles in Posterior object. Every Parameter (or VariableState) in the Posterior is updated to contain
    num_particles samples from its prior distribution, such that the parameter is now (num_particles, shape_of_param).

    Arguments:
        posterior -- GPJax posterior, should be instance of abstract posterior class.
        num_particles -- the number of particles.
        rng_key -- jax.PRNGKey for sampling the prior.

    Returns:
        _description_
    """
    def sample_prior():
        return param.prior.sample(seed=subkey, sample_shape=tuple([num_particles] + list(param.value.shape)))   # Sample from prior.
    graphdef, params, *static_state = nnx.split(posterior, nnx.VariableState, ...)    # Split Posterior nnx.Module to PyTree of params. 
    params_flat, struct = jtu.tree_flatten(params, lambda x: isinstance(x, nnx.VariableState))   # Flatten that PyTree.
    particles = []
    for param in params_flat:   # Iterate param1, param2, ... paramN
        key, subkey = jr.split(rng_key)
        param.value = sample_prior()
        param.sharding = ( "model",)
        particles.append(param)
    particle_params = jtu.tree_unflatten(struct, particles)
    particles = nnx.merge(graphdef, particle_params)
    return particles   # list of posteriors


class Particles:
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
    