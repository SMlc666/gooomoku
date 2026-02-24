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


def _apply_root_dirichlet_noise(
    prior_logits: jnp.ndarray,
    legal_mask: jnp.ndarray,
    rng_key,
    fraction: float,
    alpha: float,
) -> jnp.ndarray:
    probs = jax.nn.softmax(prior_logits, axis=-1)
    num_actions = prior_logits.shape[-1]

    noise = jax.random.dirichlet(
        rng_key,
        jnp.full((num_actions,), jnp.float32(alpha), dtype=probs.dtype),
        shape=(prior_logits.shape[0],),
    )
    legal = legal_mask.astype(probs.dtype)
    noise = noise * legal
    noise = noise / jnp.clip(jnp.sum(noise, axis=-1, keepdims=True), 1e-8)

    mixed = (jnp.float32(1.0) - jnp.float32(fraction)) * probs + jnp.float32(fraction) * noise
    mixed = mixed * legal
    mixed = mixed / jnp.clip(jnp.sum(mixed, axis=-1, keepdims=True), 1e-8)

    min_logit = jnp.finfo(prior_logits.dtype).min
    return jnp.where(legal_mask, jnp.log(jnp.clip(mixed, 1e-20)), min_logit)


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
    root_dirichlet_fraction: float = 0.0,
    root_dirichlet_alpha: float = 0.03,
):
    recurrent = functools.partial(recurrent_fn, model=model)
    use_root_noise = root_dirichlet_fraction > 0.0

    @jax.jit
    def search_fn(params, states: env.GomokuState, rng_key):
        search_key = rng_key
        root = root_output(params=params, model=model, states=states)
        invalid_actions = ~env.batch_legal_action_mask(states)
        if use_root_noise:
            noise_key, search_key = jax.random.split(rng_key)
            legal_actions = ~invalid_actions
            noisy_logits = _apply_root_dirichlet_noise(
                prior_logits=root.prior_logits,
                legal_mask=legal_actions,
                rng_key=noise_key,
                fraction=root_dirichlet_fraction,
                alpha=root_dirichlet_alpha,
            )
            root = mctx.RootFnOutput(
                prior_logits=noisy_logits,
                value=root.value,
                embedding=root.embedding,
            )
        return mctx.gumbel_muzero_policy(
            params=params,
            rng_key=search_key,
            root=root,
            recurrent_fn=recurrent,
            num_simulations=num_simulations,
            invalid_actions=invalid_actions,
            max_num_considered_actions=max_num_considered_actions,
            gumbel_scale=gumbel_scale,
        )

    return search_fn
