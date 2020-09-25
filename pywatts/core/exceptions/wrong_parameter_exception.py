class WrongParameterException(Exception):
    """
    Exception which indicates an error caused by wrong parameters
    Attributes:
        message -- explanation of the exception
        possible_solution -- Possible solution for solving the issue
        module -- the module which raised that exception
    """

    def __init__(self, message, possible_solution, module):
        self.message = message + " " + possible_solution
        self.module = module
