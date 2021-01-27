"""
This type stub file was generated by pyright.
"""

from torch.distributions.transformed_distribution import TransformedDistribution

"""
This type stub file was generated by pyright.
"""
class HalfCauchy(TransformedDistribution):
    r"""
    Creates a half-Cauchy distribution parameterized by `scale` where::

        X ~ Cauchy(0, scale)
        Y = |X| ~ HalfCauchy(scale)

    Example::

        >>> m = HalfCauchy(torch.tensor([1.0]))
        >>> m.sample()  # half-cauchy distributed with scale=1
        tensor([ 2.3214])

    Args:
        scale (float or Tensor): scale of the full Cauchy distribution
    """
    arg_constraints = ...
    support = ...
    has_rsample = ...
    def __init__(self, scale, validate_args=...) -> None:
        ...
    
    def expand(self, batch_shape, _instance=...):
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
    
    def log_prob(self, value):
        ...
    
    def cdf(self, value):
        ...
    
    def icdf(self, prob):
        ...
    
    def entropy(self):
        ...
    


