"""In-container Edward2 probe."""
import json
import pathlib

import tensorflow as tf
from tensorflow_probability import edward2 as ed


def _tensor(value):
    return tf.constant(value)


def _make_value_setter(value_map):
    def interceptor(rv_constructor, *args, **kwargs):
        name = kwargs.get("name")
        if name is not None and name in value_map:
            kwargs["value"] = tf.constant(value_map[name])
        return rv_constructor(*args, **kwargs)
    return interceptor


def main() -> None:
    io = pathlib.Path("/io")
    source = (io / "source.py").read_text()
    points = json.loads((io / "points.json").read_text())

    ns = {"edward2": ed, "ed": ed, "tf": tf}
    exec(source, ns)  # noqa: S102
    model = ns["model"]

    log_densities = []
    for pt in points:
        data_kw = {k: _tensor(v) for k, v in pt.get("data", {}).items()}
        value_map = {**pt.get("params", {}), **pt.get("data", {})}
        with ed.interception(_make_value_setter(value_map)):
            with ed.tape() as recorded:
                model(**data_kw)
        total = sum(
            rv.distribution.log_prob(rv.value) for rv in recorded.values()
        )
        log_densities.append(float(tf.reduce_sum(total).numpy()))

    (io / "result.json").write_text(
        json.dumps({"log_densities": log_densities})
    )


if __name__ == "__main__":
    main()
