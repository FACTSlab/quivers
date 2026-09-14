"""QIEC distribution bridge for generated Pyro modules.

A QIEC tensor is a nested tuple until it reaches a distribution, where
it becomes a tensor of the host's default floating dtype; a log density
is the distribution's own ``log_prob`` at the converted value.
"""


def _qvr_qiec_nested(value):
    if isinstance(value, tuple):
        return [_qvr_qiec_nested(item) for item in value]
    if isinstance(value, bool):
        return float(value)
    return value


def _qvr_qiec_array(value):
    value = _qvr_qiec_value(value)
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(_qvr_qiec_nested(value), dtype=torch.get_default_dtype())


def _qvr_qiec_log_density(distribution, value):
    return distribution.log_prob(_qvr_qiec_array(value))
