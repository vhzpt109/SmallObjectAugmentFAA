from torch.utils.data import Sampler


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


DOTALABELS = {
    "plane": 1,
    "ship": 2,
    "storage-tank": 3,
    "baseball-diamond": 4,
    "tennis-court": 5,
    "basketball-court": 6,
    "ground-track-field": 7,
    "harbor": 8,
    "bridge": 9,
    "large-vehicle": 10,
    "small-vehicle": 11,
    "helicopter": 12,
    "roundabout": 13,
    "soccer-ball-field": 14,
    "swimming-pool": 15,
    "container-crane": 16,
    "airport": 17,
    "helipad": 18,
}