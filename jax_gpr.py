# Enable Float64 for more stable matrix inversions.
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jaxtyping import install_import_hook
from jax import tree_util as jtu
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax.distributions as tfd
import blackjax
from blackjax.smc import resampling

from jax_gpr_particles import init_particles, log_likelihood, log_prior

config.update("jax_enable_x64", True)


with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx


key = jr.key(12)

cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
n = 100
noise = 0.3

key, subkey = jr.split(key)
x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,)).reshape(-1, 1)
def f(x): return jnp.sin(4 * x) + jnp.cos(2 * x)


signal = f(x)
y = signal + jr.normal(subkey, shape=signal.shape) * noise

D = gpx.Dataset(X=x, y=y)

xtest = jnp.linspace(-3.5, 3.5, 500).reshape(-1, 1)
ytest = f(xtest)

kernel = gpx.kernels.RBF()  # 1-dimensional input

meanf = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)

posterior = prior * likelihood
# Set prior distributions on kernel parameters.
graphdef, params, * \
    static_state = nnx.split(posterior, gpx.parameters.Parameter, ...)
params_bijection = gpx.parameters.DEFAULT_BIJECTION

params = gpx.parameters.set_priors(params, param_priors={"prior": {"kernel": {"variance": tfd.Normal(1, 1), 
                                                                              "lengthscale": tfd.Normal(1, 1)},
                                                                   "mean_function": {"constant": tfd.Normal(0, 1)}},
                                                         "likelihood": {"obs_stddev": tfd.Normal(1, 1)}})

posterior = nnx.merge(graphdef, params)

flattened, _ = jtu.tree_flatten_with_path(params)
for key_path, value in flattened:
    print(f'Value of tree{jtu.keystr(key_path)}: {value}')

particles = init_particles(key, posterior, 10)

def loglikelihood(p):
    return log_likelihood(D, p)

def prior_log_prob():
    return particles.log_prior()

print(prior_log_prob())

# hmc_parameters = dict(
#     step_size=1e-4, inverse_mass_matrix=jnp.eye(8), num_integration_steps=1
# )

# def smc_inference_loop(rng_key, smc_kernel, initial_state):
#     """Run the temepered SMC algorithm.

#     We run the adaptive algorithm until the tempering parameter lambda reaches the value
#     lambda=1.

#     """

#     def cond(carry):
#         i, state, _k = carry
#         return state.lmbda < 1

#     def one_step(carry):
#         i, state, k = carry
#         k, subk = jax.random.split(k, 2)
#         state, _ = smc_kernel(subk, state)
#         return i + 1, state, k

#     n_iter, final_state, _ = jax.lax.while_loop(
#         cond, one_step, (0, initial_state, rng_key)
#     )

#     return n_iter, final_state


# tempered = blackjax.adaptive_tempered_smc(
#     prior_log_prob,
#     loglikelihood,
#     blackjax.hmc.build_kernel(),
#     blackjax.hmc.init,
#     hmc_parameters,
#     resampling.systematic,
#     0.5,
#     num_mcmc_steps=1,
# )

# rng_key, init_key, sample_key = jax.random.split(key, 3)
# graphdef, params, * \
#     static_state = nnx.split(particles, gpx.parameters.Parameter, ...)
# params_bijection = gpx.parameters.DEFAULT_BIJECTION
# initial_smc_state = params
# initial_smc_state = tempered.init(initial_smc_state)

# n_iter, smc_samples = smc_inference_loop(sample_key, tempered.step, initial_smc_state)
# print("Number of steps in the adaptive algorithm: ", n_iter.item())