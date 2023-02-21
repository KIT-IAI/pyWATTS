class InvalidInputException(Exception):
    """
    Exception which indicates that the requested input for a module is invalid
    Attributes:
        message -- explanation of the exception
    """

    def __init__(self, message):
        self.message = message
