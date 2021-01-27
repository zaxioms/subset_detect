"""
This type stub file was generated by pyright.
"""

from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution

"""
This type stub file was generated by pyright.
"""
class Pareto(TransformedDistribution):
    r"""
    Samples from a Pareto Type 1 distribution.

    Example::

        >>> m = Pareto(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Pareto distribution with scale=1 and alpha=1
        tensor([ 1.5623])

    Args:
        scale (float or Tensor): Scale parameter of the distribution
        alpha (float or Tensor): Shape parameter of the distribution
    """
    arg_constraints = ...
    def __init__(self, scale, alpha, validate_args=...) -> None:
        ...
    
    def expand(self, batch_shape, _instance=...):
        ...
    
    @property
    def mean(self):
        ...
    
    @property
    def variance(self):
        ...
    
    @constraints.dependent_property
    def support(self):
        ...
    
    def entropy(self):
        ...
    


