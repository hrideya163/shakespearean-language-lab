import random


def bootstrap_ci(metric, n_items, n_resamples=1000, seed=0):
    rng = random.Random(seed)
    values = sorted(metric([rng.randrange(n_items) for _ in range(n_items)])
                    for _ in range(n_resamples))
    return values[int(0.025 * n_resamples)], values[int(0.975 * n_resamples) - 1]
