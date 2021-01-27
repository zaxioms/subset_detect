"""
This type stub file was generated by pyright.
"""

import sys

"""
This type stub file was generated by pyright.
"""
_supports_context = sys.version_info >= (3, 4)
class ProcessContext:
    def __init__(self, processes, error_queues) -> None:
        ...
    
    def pids(self):
        ...
    
    def join(self, timeout=...):
        r"""
        Tries to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes and raises an exception with the cause
        of the first process exiting.

        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.

        Arguments:
            timeout (float): Wait this long before giving up on waiting.
        """
        ...
    


class SpawnContext(ProcessContext):
    def __init__(self, processes, error_queues) -> None:
        ...
    


def start_processes(fn, args=..., nprocs=..., join=..., daemon=..., start_method=...):
    ...

def spawn(fn, args=..., nprocs=..., join=..., daemon=..., start_method=...):
    r"""Spawns ``nprocs`` processes that run ``fn`` with ``args``.

    If one of the processes exits with a non-zero exit status, the
    remaining processes are killed and an exception is raised with the
    cause of termination. In the case an exception was caught in the
    child process, it is forwarded and its traceback is included in
    the exception raised in the parent process.

    Arguments:
        fn (function): Function is called as the entrypoint of the
            spawned process. This function must be defined at the top
            level of a module so it can be pickled and spawned. This
            is a requirement imposed by multiprocessing.

            The function is called as ``fn(i, *args)``, where ``i`` is
            the process index and ``args`` is the passed through tuple
            of arguments.

        args (tuple): Arguments passed to ``fn``.
        nprocs (int): Number of processes to spawn.
        join (bool): Perform a blocking join on all processes.
        daemon (bool): The spawned processes' daemon flag. If set to True,
                       daemonic processes will be created.
        start_method (string): (deprecated) this method will always use ``spawn``
                               as the start method. To use a different start method
                               use ``start_processes()``.

    Returns:
        None if ``join`` is ``True``,
        :class:`~ProcessContext` if ``join`` is ``False``

    """
    ...

