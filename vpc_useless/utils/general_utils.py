import math
import numpy as np


def cycle(iterable):
    '''
    Itertools.cycle tries to save all inputs, which causes memory usage to grow. 
    See: https://github.com/pytorch/pytorch/issues/23900
    '''
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)



def get_powers_of_two(n, force_last=True):
    '''
    Returns an array of all powers of two, smaller or equal to n.

    Args:
        n: Number for which to calculate all smaller powers of two.
        force_last: Even if n is not a power of two, set to True to include it.

    Returns:
        Numpy array containing the powers of two in ascending order.
    '''
    largest_pow2 = 2 ** int(math.log(n, 2)) # Largest power of two smaller or equal to n
    powers = np.geomspace(1, largest_pow2, (np.log2(largest_pow2) + 1).astype(int)).round().astype(int)
    powers = np.append(powers, n) if n != largest_pow2 and force_last else powers
    return powers

def domain_losses(channel_losses, dataset, suffix):
    domain_loss_dict = {}
    loss_channel_idx = 0
    for domain_id, domain_channels in zip(dataset.domain_ids, dataset.domain_channels):
        for n in range(domain_channels):
            domain_loss_dict['{}_{}_{}'.format(domain_id, n, suffix)] = channel_losses[loss_channel_idx].item()
            loss_channel_idx += 1
    return domain_loss_dict
