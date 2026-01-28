import itertools
import torch

from reg23_experiments.registration.lib import similarity_metric

def test_local_ncc():
    size = 12
    a = torch.empty((size, size), dtype=torch.float32)
    for j, i in itertools.product(range(size), range(size)):
        a[j, i] = float(j * size + i)
    b = a.clone()
    b[:size // 2, :] *= -1.0
    ncc = similarity_metric.local_ncc(a, b, kernel_size=4)
    print(ncc)