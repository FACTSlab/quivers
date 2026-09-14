"""QIEC distribution bridge for generated PyMC modules.

A QIEC tensor is a nested tuple until it reaches a distribution, where
it becomes a NumPy array; a log density is ``pymc.logp`` of the
unregistered distribution at the converted value, a PyTensor expression
the enclosing model evaluates. A family whose PyMC support is shifted
against the QIEC convention is wrapped with its offset, and the offset
is added to the scored value.
"""


class _QvrQiecShifted:
    def __init__(self, distribution, offset):
        self.distribution = distribution
        self.offset = offset


def _qvr_qiec_shifted(distribution, offset):
    return _QvrQiecShifted(distribution, offset)


def _qvr_qiec_nested(value):
    if isinstance(value, tuple):
        return [_qvr_qiec_nested(item) for item in value]
    if isinstance(value, bool):
        return float(value)
    return value


def _qvr_qiec_array(value):
    value = _qvr_qiec_value(value)
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, tuple):
        return np.asarray(_qvr_qiec_nested(value), dtype="float64")
    return _qvr_qiec_nested(value)


def _qvr_qiec_log_density(distribution, value):
    point = _qvr_qiec_array(value)
    if isinstance(distribution, _QvrQiecShifted):
        return pymc.logp(distribution.distribution, point + distribution.offset)
    return pymc.logp(distribution, point)
