"""
This type stub file was generated by pyright.
"""

from torch.distributions.transformed_distribution import TransformedDistribution

"""
This type stub file was generated by pyright.
"""
class LogNormal(TransformedDistribution):
    r"""
    Creates a log-normal distribution parameterized by
    :attr:`loc` and :attr:`scale` where::

        X ~ Normal(loc, scale)
        Y = exp(X) ~ LogNormal(loc, scale)

    Example::

        >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of log of distribution
        scale (float or Tensor): standard deviation of log of the distribution
    """
    arg_constraints = ...
    support = ...
    has_rsample = ...
    def __init__(self, loc, scale, validate_args=...) -> None:
        ...
    
    def expand(self, batch_shape, _instance=...):
        ...
    
    @property
    def loc(self):
        ...
    
    @property
    def scale(self):
        ...
    
    @property
    def mean(self):
        ...
    
    @property
    def variance(self):
        ...
    
    def entropy(self):
        ...
    

