"""
This type stub file was generated by pyright.
"""

from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property

"""
This type stub file was generated by pyright.
"""
class LowRankMultivariateNormal(Distribution):
    r"""
    Creates a multivariate normal distribution with covariance matrix having a low-rank form
    parameterized by :attr:`cov_factor` and :attr:`cov_diag`::

        covariance_matrix = cov_factor @ cov_factor.T + cov_diag

    Example:

        >>> m = LowRankMultivariateNormal(torch.zeros(2), torch.tensor([[1.], [0.]]), torch.ones(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]`, cov_factor=`[[1],[0]]`, cov_diag=`[1,1]`
        tensor([-0.2102, -0.5429])

    Args:
        loc (Tensor): mean of the distribution with shape `batch_shape + event_shape`
        cov_factor (Tensor): factor part of low-rank form of covariance matrix with shape
            `batch_shape + event_shape + (rank,)`
        cov_diag (Tensor): diagonal part of low-rank form of covariance matrix with shape
            `batch_shape + event_shape`

    Note:
        The computation for determinant and inverse of covariance matrix is avoided when
        `cov_factor.shape[1] << cov_factor.shape[0]` thanks to `Woodbury matrix identity
        <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_ and
        `matrix determinant lemma <https://en.wikipedia.org/wiki/Matrix_determinant_lemma>`_.
        Thanks to these formulas, we just need to compute the determinant and inverse of
        the small size "capacitance" matrix::

            capacitance = I + cov_factor.T @ inv(cov_diag) @ cov_factor
    """
    arg_constraints = ...
    support = ...
    has_rsample = ...
    def __init__(self, loc, cov_factor, cov_diag, validate_args=...) -> None:
        ...
    
    def expand(self, batch_shape, _instance=...):
        ...
    
    @property
    def mean(self):
        ...
    
    @lazy_property
    def variance(self):
        ...
    
    @lazy_property
    def scale_tril(self):
        ...
    
    @lazy_property
    def covariance_matrix(self):
        ...
    
    @lazy_property
    def precision_matrix(self):
        ...
    
    def rsample(self, sample_shape=...):
        ...
    
    def log_prob(self, value):
        ...
    
    def entropy(self):
        ...
    


