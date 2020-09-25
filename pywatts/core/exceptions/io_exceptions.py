class IOException(Exception):
    """
    Exception which is raised by IO related tasks
    Attributes:
        message -- explanation of the exception
    """

    def __init__(self, message):
        self.message = message
