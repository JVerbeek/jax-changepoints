# Enable Float64 for more stable matrix inversions.
import sys
sys.path.append("/home/janneke/jax-smc/GPJax")
from jax import config
import jax.scipy as jsp
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
    install_import_hook
)

import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax as tfp
import beartype.typing as tp
from flax import nnx
#from examples.utils import use_mpl_style
from gpjax.kernels.computations import DenseKernelComputation
from gpjax.parameters import (
    DEFAULT_BIJECTION,
    PositiveReal,
    Static,
    Real,
)


with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx


tfb = tfp.bijectors

key = jr.key(42)

class ChangePoints(gpx.kernels.AbstractKernel):
    locations: Real
    steepness: Real
    
    def __init__(
        self,
        kernels,
        locations: tp.Union[Float, nnx.Variable[Array]] = 1.0,
        steepness: tp.Union[Float, nnx.Variable[Array]] = 1.0,
        active_dims: list[int] | slice | None = None,
        n_dims: int | None = None,
    ):
        super().__init__(active_dims, n_dims, DenseKernelComputation())
        assert len(locations) == len(steepness), "Locations and steepnesses must be of equal length"
        assert len(locations) == len(kernels) - 1, "Number of locations must be one less than the number of kernels"
        assert len(steepness) == len(kernels) - 1, "Number of steepnesses must be one less than the number of kernels"
        argsort_locs = jnp.argsort(jnp.array(locations))  # Ensure locations and steepnesses have same sorting
        self.locations = Real(jnp.array(locations)[argsort_locs], tag="locations")
        self.steepness = Real(jnp.array(steepness)[argsort_locs], tag="steepness")
        self.kernels = kernels   # List seems to work fine for now, but if we want to do something high performance this maybe needs to change?
        self.name = "ChangePoints"

    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        
        sig = jsp.special.expit(self.steepness*(x - self.locations))
        sig_ = jsp.special.expit(self.steepness*(y - self.locations)).reshape(1, -1)
        
        sig_start = sig * sig_
        sig_stop = (1 - sig) * (1 - sig_)
        starters = jnp.concatenate([jnp.expand_dims(jnp.ones(sig_start.shape[:-1]), axis=1), sig_start], axis=-1)
        stoppers = jnp.concatenate([sig_stop, jnp.expand_dims(jnp.ones(sig_stop.shape[:-1]), axis=-1)], axis=-1)
        kernel_stack = jnp.stack([k(x, y) for k in self.kernels], axis=-1)
        K = jnp.sum(kernel_stack * starters * stoppers, axis=-1)
        return K.squeeze()
    
if __name__=="__main__":
    x = jnp.linspace(-20.0, 20.0, num=200).reshape(-1, 1)

    #One changepoint
    cpk = ChangePoints(kernels=[gpx.kernels.RBF(), gpx.kernels.RBF()], locations=[0], steepness=[5]) 

    fig, ax = plt.subplots(ncols=1, figsize=(3, 3))
    im0 = ax.matshow(cpk.gram(x).to_dense())
    
    plt.show()
    
    # Multiple changepoints  
    cpk = ChangePoints(kernels=[gpx.kernels.RBF(), gpx.kernels.RBF(), gpx.kernels.RBF()], locations=[10, 0], steepness=[5, 10])  
    fig, ax = plt.subplots(ncols=1, figsize=(3, 3))
    im0 = ax.matshow(cpk.gram(x).to_dense())
    ax.set_xticks(jnp.linspace(0, 200, 21), jnp.linspace(-20, 20, 21).round(2))
    ax.set_yticks(jnp.linspace(0, 200, 21), jnp.linspace(-20, 20, 21).round(2))
    plt.show()
    
    kernels = [ChangePoints(kernels=[gpx.kernels.RBF(), gpx.kernels.Matern12(), gpx.kernels.RBF()], locations=[-10, 10], steepness=[5, 10])]
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(7, 6), tight_layout=True)

    meanf = gpx.mean_functions.Zero()

    for k, ax in zip(kernels, [axes]):
        prior = gpx.gps.Prior(mean_function=meanf, kernel=k)
        rv = prior(x)
        y = rv.sample(seed=key, sample_shape=(3,))
        ax.plot(x, y.T, alpha=0.7)
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        ax.set_title(k.name)
        [ax.axvline(l, color="red", linestyle=":") for l in k.locations.value.tolist()]
    plt.show()