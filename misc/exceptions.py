# -*- coding: utf8 -*-

class WrongArgumentsError(Exception):
    """
    The program was called with incorrect arguments or an incorrect combination of them.
    """
    pass

class WrongShapeError(Exception):
    """
    A sequence has the wrong shape.
    """
    pass

class ClassNotRegisteredError(Exception):
    """
    Tried to create an environment or agent instance that is not registered.
    """
    pass
