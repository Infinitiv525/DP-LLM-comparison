"""Independent float64 finite-difference diagnostics on normalized coordinates."""
import numpy as np


def differential(value, reynolds):
    u, v, p = np.moveaxis(np.asarray(value, dtype=np.float64), -1, 0)
    h = 1 / u.shape[0]
    dx = lambda a: (np.roll(a, -1, 1) - np.roll(a, 1, 1)) / (2 * h)
    dy = lambda a: (np.roll(a, -1, 0) - np.roll(a, 1, 0)) / (2 * h)
    lap = lambda a: (np.roll(a, -1, 1) + np.roll(a, 1, 1)
                     + np.roll(a, -1, 0) + np.roll(a, 1, 0) - 4 * a) / h**2
    advection = np.stack((u * dx(u) + v * dy(u), u * dx(v) + v * dy(v)), -1)
    pressure = np.stack((dx(p), dy(p)), -1)
    diffusion = np.stack((lap(u), lap(v)), -1) / reynolds
    residual = np.concatenate((advection + pressure - diffusion, (dx(u) + dy(v))[..., None]), -1)
    return residual, advection, pressure, diffusion


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Finite nonempty diagnostics required')
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(values.max()))


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64)**2)))


def vector_error(prediction, target):
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    speed = np.linalg.norm(delta, axis=-1)
    return dict(component_rms=rms(delta), velocity_error_rms=float(np.sqrt(np.mean(speed**2))),
                point_error_p95=float(np.percentile(speed, 95)), maximum_point_error=float(speed.max()))
