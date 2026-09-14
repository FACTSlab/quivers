"""QIEC distribution bridge for generated Edward2 modules.

A QIEC tensor is a nested tuple until it reaches a distribution, where
it becomes a ``float32`` TensorFlow tensor; a log density is the
TensorFlow Probability distribution's own ``log_prob`` at the converted
value.
"""


def _qvr_qiec_nested(value):
    if isinstance(value, tuple):
        return [_qvr_qiec_nested(item) for item in value]
    if isinstance(value, bool):
        return float(value)
    return value


def _qvr_qiec_array(value):
    value = _qvr_qiec_value(value)
    if isinstance(value, tf.Tensor):
        return value
    return tf.convert_to_tensor(_qvr_qiec_nested(value), dtype=tf.float32)


def _qvr_qiec_log_density(distribution, value):
    return distribution.log_prob(_qvr_qiec_array(value))
