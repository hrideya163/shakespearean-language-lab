"""Bootstrap confidence intervals: how sure are we about a number measured on a test set?

Idea: the test set is only a SAMPLE of all possible sentences. We pretend to draw new
test sets by picking sentences at random WITH replacement, recompute the metric each
time, and look at how much it moves. The middle 95% of those values is the
95% confidence interval. If the intervals of two systems overlap a lot, the
difference between them may just be luck.
"""
import random


def bootstrap_ci(metric, n_items, n_resamples=1000, seed=0):
    """metric(indices) -> number, computed on the chosen items.
    Returns (low, high) of the 95% confidence interval."""
    rng = random.Random(seed)
    values = sorted(metric([rng.randrange(n_items) for _ in range(n_items)])
                    for _ in range(n_resamples))
    return values[int(0.025 * n_resamples)], values[int(0.975 * n_resamples) - 1]
