import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
from jax import Array
import jax
from jax.flatten_util import ravel_pytree

import dLux as dl
import dLux.utils as dlu

import zodiax as zdx
import equinox as eqx
import optax
from zodiax import optimisation as opt
import optimistix as optx
from tqdm.auto import tqdm

from apertures import *
from detectors import *
from spectra import *
from models import *
from stats import *

"""
fitting utilities
"""

def get_optimiser_new(model_params, optimisers):
    param_spec = ModelParams({param: param for param in model_params.keys()})
    optim = optax.multi_transform(optimisers, param_spec)
    return optim, optim.init(model_params)

def loss_fn(params, exposures, model):
    mdl = params.inject(model)
    return np.nansum(np.asarray([posterior(mdl,exposure) for exposure in exposures]))

def optimise_optimistix(params, model, exposures, project=True, diag=False, nbatches=None):
    if not nbatches:
        nbatches=len(exposures)*5
    if project:
        f = lambda params: loss_fn(params, exposures, model)
        F, unflatten = zdx.hessian(f, ModelParams(params), nbatches=nbatches, checkpoint=True)
        if diag:
            F = np.diag(np.diag(F))
            

    def projected_loss_fn(u, args):
        exposures, model, project_fn = args
        params = project_fn(u)
        return loss_fn(params, exposures, model)

    # Estimate our initial parameters from the data
    params = ModelParams(params)
    X0, unravel = ravel_pytree(params)

    # Generate the projection matrix P, projection function, and initial vector
    P = zdx.optimisation.eigen_projection(fmat=F) if project else np.eye(X0.shape[0])
    project_fn = lambda u: unravel(X0 + np.dot(P, u))
    X = np.zeros(P.shape[-1])


    # Minimise algorithm
    args = (exposures, model, project_fn)
    solver = optx.BestSoFarMinimiser(optx.LBFGS(rtol=1e-6, atol=1e-6))
    sol = optx.minimise(projected_loss_fn, solver, X, args, max_steps=1024, throw=False)
    return project_fn(sol.value)

def optimise_new(params, model, exposures, optimisers, epochs, diag=True, nbatches=1, use_c=False, return_c=False):

    if use_c is not False:
        C = use_c
    else:
        f = lambda params: loss_fn(ModelParams(params), exposures, model)
        F, unflatten = zdx.hessian(f, params, nbatches=nbatches, checkpoint=True)

        if diag:
            C = dlu.nandiv(1, np.abs((np.diag(F))), fill=0.)
        else:
            C = np.linalg.inv(F)
        
    optim, state = opt.map_optimisers(params, optimisers)

    loss_grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(lambda params, exposures, model: loss_fn(ModelParams(params), exposures, model)))

    pbar = tqdm(range(epochs))
    losses, params_history = [], []
    for step in pbar:
        loss, grads = loss_grad_fn(params, exposures, model)

        # Normalise the gradients by the fisher matrix to get a natural gradient step
        G, unflatten = ravel_pytree(grads)
        if diag:
            grads = unflatten(G*C)
        else:
            grads = unflatten(np.dot(G, C))

        updates, state = optim.update(grads, state)
        params = optax.apply_updates(params, updates)
        pbar.set_postfix(log_loss=f"{np.log10(loss):.4f}")
        losses.append(loss)
        params_history.append(params)
    losses = np.array(losses)

    if return_c:
        return losses, params_history, C

    return losses, params_history
