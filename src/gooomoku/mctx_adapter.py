from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import mctx

from gooomoku import env
from gooomoku.net import PolicyValueNet


def _masked_logits(logits: jnp.ndarray, legal_mask: jnp.ndarray) -> jnp.ndarray:
    min_logit = jnp.finfo(logits.dtype).min
    return jnp.where(legal_mask, logits, min_logit)


def root_output(params, model: PolicyValueNet, states: env.GomokuState) -> mctx.RootFnOutput:
    obs = env.batch_encode_states(states)
    logits, value = model.apply(params, obs)
    legal = env.batch_legal_action_mask(states)
    prior_logits = _masked_logits(logits, legal)
    return mctx.RootFnOutput(prior_logits=prior_logits, value=value, embedding=states)


def recurrent_fn(params, rng_key, action, embedding: env.GomokuState, *, model: PolicyValueNet):
    del rng_key
    next_state, reward, done = env.batch_step(embedding, action)
    obs = env.batch_encode_states(next_state)
    logits, value = model.apply(params, obs)
    legal = env.batch_legal_action_mask(next_state)
    prior_logits = _masked_logits(logits, legal)
    discount = jnp.where(done, jnp.float32(0.0), jnp.float32(1.0))
    output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=prior_logits,
        value=value,
    )
    return output, next_state


def run_gumbel_search(
    *,
    params,
    model: PolicyValueNet,
    states: env.GomokuState,
    rng_key,
    num_simulations: int,
    max_num_considered_actions: int,
    gumbel_scale: float = 1.0,
):
    root = root_output(params=params, model=model, states=states)
    invalid_actions = ~env.batch_legal_action_mask(states)
    return mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=functools.partial(recurrent_fn, model=model),
        num_simulations=num_simulations,
        invalid_actions=invalid_actions,
        max_num_considered_actions=max_num_considered_actions,
        gumbel_scale=gumbel_scale,
    )


def build_search_fn(
    *,
    model: PolicyValueNet,
    num_simulations: int,
    max_num_considered_actions: int,
    gumbel_scale: float = 1.0,
):
    recurrent = functools.partial(recurrent_fn, model=model)

    @jax.jit
    def search_fn(params, states: env.GomokuState, rng_key):
        root = root_output(params=params, model=model, states=states)
        invalid_actions = ~env.batch_legal_action_mask(states)
        return mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent,
            num_simulations=num_simulations,
            invalid_actions=invalid_actions,
            max_num_considered_actions=max_num_considered_actions,
            gumbel_scale=gumbel_scale,
        )

    return search_fn
