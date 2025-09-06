import torch

class Helpers:
    def __init__(self):
        pass
    # helper function
    def exists(self, x):
        return x is not None

    def noop(self, *args, **kwargs):
        pass

    def is_odd(self, n):
        return (n % 2) == 1

    def default(self, val, d):
        if self.exists(val):
            return val
        return d() if callable(d) else d

    def cycle(self, dl):
        while True:
            for data in dl:
                yield data

    def num_to_groups(self, num, divisor):
        groups = num // divisor
        remainder = num % divisor
        arr = [divisor] * groups
        if remainder > 0:
            arr.append(remainder)
        return arr

    def prob_mask_like(self, shape, prob, device):
        if prob == 1:
            return torch.ones(shape, device=device, dtype=torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device=device, dtype=torch.bool)
        else:
            return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

    def is_list_str(self, x):
        if not isinstance(x, list, tuple):
            return False
        return all([type(el) == str for el in x])
