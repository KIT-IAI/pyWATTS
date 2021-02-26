class StepCreationException(Exception):
    """
    Exception which occurs during the creation of steps and adding modules to the pipeline.
    Attributes:
        message -- explanation of the exception
        module -- the module which raised that exception
    """

    def __init__(self, message, module=""):
        self.message = message
        self.module = module
