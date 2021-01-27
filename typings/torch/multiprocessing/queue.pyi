"""
This type stub file was generated by pyright.
"""

import multiprocessing

class ConnectionWrapper(object):
    """Proxy class for _multiprocessing.Connection which uses ForkingPickler to
    serialize objects"""
    def __init__(self, conn) -> None:
        ...
    
    def send(self, obj):
        ...
    
    def recv(self):
        ...
    
    def __getattr__(self, name):
        ...
    


class Queue(multiprocessing.queues.Queue):
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class SimpleQueue(multiprocessing.queues.SimpleQueue):
    ...


