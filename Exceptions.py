#!/usr/bin/env python
# -*- coding: utf8 -*-

class WrongArgumentsException(Exception):
    """
    The program was called with incorrect arguments or an incorrect combination of them.
    """
    pass

class WrongShapeException(Exception):
    """
    A sequence has the wrong shape.
    """
    pass

class ClassNotRegisteredException(Exception):
    """
    Tried to create an environment that is not registered.
    """
    pass
