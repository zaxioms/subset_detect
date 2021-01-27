"""
This type stub file was generated by pyright.
"""

import re

"""
This type stub file was generated by pyright.
"""
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
MASTER_BRANCH = 'master'
ENV_TORCH_HOME = 'TORCH_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'
VAR_DEPENDENCY = 'dependencies'
MODULE_HUBCONF = 'hubconf.py'
READ_DATA_CHUNK = 8192
_hub_dir = None
def import_module(name, path):
    ...

def get_dir():
    r"""
    Get the Torch Hub cache directory used for storing downloaded models & weights.

    If :func:`~torch.hub.set_dir` is not called, default path is ``$TORCH_HOME/hub`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesystem layout, with a default value ``~/.cache`` if the environment
    variable is not set.
    """
    ...

def set_dir(d):
    r"""
    Optionally set the Torch Hub directory used to save downloaded models & weights.

    Args:
        d (string): path to a local folder to save downloaded models & weights.
    """
    ...

def list(github, force_reload=...):
    r"""
    List all entrypoints available in `github` hubconf.

    Args:
        github (string): a string with format "repo_owner/repo_name[:tag_name]" with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'pytorch/vision[:hub]'
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.
            Default is `False`.
    Returns:
        entrypoints: a list of available entrypoint names

    Example:
        >>> entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
    """
    ...

def help(github, model, force_reload=...):
    r"""
    Show the docstring of entrypoint `model`.

    Args:
        github (string): a string with format <repo_owner/repo_name[:tag_name]> with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'pytorch/vision[:hub]'
        model (string): a string of entrypoint name defined in repo's hubconf.py
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.
            Default is `False`.
    Example:
        >>> print(torch.hub.help('pytorch/vision', 'resnet18', force_reload=True))
    """
    ...

def load(repo_or_dir, model, *args, **kwargs):
    r"""
    Load a model from a github repo or a local directory.

    Note: Loading a model is the typical use case, but this can also be used to
    for loading other objects such as tokenizers, loss functions, etc.

    If :attr:`source` is ``'github'``, :attr:`repo_or_dir` is expected to be
    of the form ``repo_owner/repo_name[:tag_name]`` with an optional
    tag/branch.

    If :attr:`source` is ``'local'``, :attr:`repo_or_dir` is expected to be a
    path to a local directory.

    Args:
        repo_or_dir (string): repo name (``repo_owner/repo_name[:tag_name]``),
            if ``source = 'github'``; or a path to a local directory, if
            ``source = 'local'``.
        model (string): the name of a callable (entrypoint) defined in the
            repo/dir's ``hubconf.py``.
        *args (optional): the corresponding args for callable :attr:`model`.
        source (string, optional): ``'github'`` | ``'local'``. Specifies how
            ``repo_or_dir`` is to be interpreted. Default is ``'github'``.
        force_reload (bool, optional): whether to force a fresh download of
            the github repo unconditionally. Does not have any effect if
            ``source = 'local'``. Default is ``False``.
        verbose (bool, optional): If ``False``, mute messages about hitting
            local caches. Note that the message about first download cannot be
            muted. Does not have any effect if ``source = 'local'``.
            Default is ``True``.
        **kwargs (optional): the corresponding kwargs for callable
            :attr:`model`.

    Returns:
        The output of the :attr:`model` callable when called with the given
        ``*args`` and ``**kwargs``.

    Example:
        >>> # from a github repo
        >>> repo = 'pytorch/vision'
        >>> model = torch.hub.load(repo, 'resnet50', pretrained=True)
        >>> # from a local directory
        >>> path = '/some/local/path/pytorch/vision'
        >>> model = torch.hub.load(path, 'resnet50', pretrained=True)
    """
    ...

def download_url_to_file(url, dst, hash_prefix=..., progress=...):
    r"""Download object at the given URL to a local path.

    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with `hash_prefix`.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Example:
        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')

    """
    ...

def load_state_dict_from_url(url, model_dir=..., map_location=..., progress=..., check_hash=..., file_name=...):
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of `model_dir` is ``<hub_dir>/checkpoints`` where
    `hub_dir` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (string, optional): name for the downloaded file. Filename from `url` will be used if not set.

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    ...

