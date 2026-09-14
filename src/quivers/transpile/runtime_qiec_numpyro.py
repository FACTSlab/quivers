"""QIEC distribution bridge for generated NumPyro modules.

A QIEC tensor is a nested tuple until it reaches a distribution, where
it becomes a floating JAX array; a log density is the distribution's own
``log_prob`` at the scored point, which keeps an integral dtype so a
discrete family can index its probabilities with it.
"""


def _qvr_qiec_nested(value):
    if isinstance(value, tuple):
        return [_qvr_qiec_nested(item) for item in value]
    if isinstance(value, bool):
        return float(value)
    return value


def _qvr_qiec_array(value):
    value = _qvr_qiec_value(value)
    if isinstance(value, jnp.ndarray):
        return value
    return jnp.asarray(_qvr_qiec_nested(value), dtype=jnp.result_type(float))


def _qvr_qiec_point(value):
    value = _qvr_qiec_value(value)
    if isinstance(value, jnp.ndarray):
        return value
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, tuple):
        return jnp.asarray(_qvr_qiec_nested(value))
    return value


def _qvr_qiec_log_density(distribution, value):
    return distribution.log_prob(_qvr_qiec_point(value))
