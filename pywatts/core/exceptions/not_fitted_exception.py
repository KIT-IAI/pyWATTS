class NotFittedException(Exception):
    """
    Exception which indicates an error caused by wrong parameters
    Attributes:
        message -- explanation of the exception
        step -- the step which raised that exception
        module -- the module which raised that exception

    """

    def __init__(self, message, step: str, module: str):
        self.message = message
        self.module_name = module
        self.step_name = step
