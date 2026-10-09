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

def optimise_optimistix(
    params,
    model,
    exposures,
    project=True,
    diag=False,
    nbatches=None,
    scales=None,
    max_steps=1024,
    progress=True,
    progress_desc="BFGS",
):

    if nbatches is None:
        nbatches = len(exposures) * 5

    # ==================================================
    # Fisher / Hessian projection
    # ==================================================

    if project:

        f = lambda params: loss_fn(
            params,
            exposures,
            model
        )

        F, unflatten = zdx.hessian(
            f,
            ModelParams(params),
            nbatches=nbatches,
            checkpoint=True
        )

        if diag:
            F = np.diag(np.diag(F))


    def projected_loss_fn(u, args):

        exposures, model, project_fn = args

        params = project_fn(u)

        return loss_fn(
            params,
            exposures,
            model
        )


    # ==================================================
    # Flatten starting parameters
    # ==================================================

    params = ModelParams(params)

    X0, unravel = ravel_pytree(params)


    # ==================================================
    # Projection matrix
    # ==================================================

    if project:

        P = zdx.optimisation.eigen_projection(
            fmat=F
        )

    else:

        P = np.eye(X0.shape[0])


    # ==================================================
    # Parameter scaling
    # ==================================================

    if scales is None:

        scale_vector = np.ones_like(X0)

    else:

        scale_vector, _ = ravel_pytree(
            ModelParams(scales)
        )

        if scale_vector.shape != X0.shape:
            raise ValueError(
                "Scale tree does not match parameter tree."
            )


    # Convert the dimensionless optimiser coordinates
    # back into the physical model parameters.
    project_fn = lambda u: unravel(
        X0 + scale_vector * np.dot(P, u)
    )

    X = np.zeros(P.shape[-1])


    # ==================================================
    # Progress bar
    # ==================================================

    if progress:

        pbar = tqdm(
            total=max_steps,
            desc=progress_desc,
            unit="step"
        )

        def progress_callback(**kwargs):

            # Optimistix calls this once for each solver step
            pbar.update(1)

            # Find the current loss supplied by Optimistix
            for item in kwargs.values():

                if isinstance(item, tuple) and len(item) == 2:

                    label, value = item

                    if label == "Loss on this step":

                        try:
                            loss_value = float(value)

                            pbar.set_postfix(
                                loss=f"{loss_value:.4e}"
                            )

                        except (TypeError, ValueError):
                            pass

    else:

        pbar = None
        progress_callback = False


    # ==================================================
    # LBFGS
    # ==================================================

    args = (
        exposures,
        model,
        project_fn
    )

    solver = optx.BestSoFarMinimiser(
        optx.LBFGS(
            rtol=1e-6,
            atol=1e-6,
            verbose=progress_callback
        )
    )

    try:

        sol = optx.minimise(
            projected_loss_fn,
            solver,
            X,
            args,
            max_steps=max_steps,
            throw=False
        )

    finally:

        if pbar is not None:
            pbar.close()


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
