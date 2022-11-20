import numpy as np

def gen_random_indices(max_random_num, min_random_num=0, random_ratio=1., random_num=True, size=0):
    # generate x% of max_random_num random indices
    pb = np.random.rand() if random_num else 1.
    num_indices = np.floor(pb*random_ratio*(max_random_num-min_random_num)) if not size else size
    indices = np.random.choice(np.arange(min_random_num, max_random_num), replace=False, size=int(num_indices))
    return np.sort(indices)