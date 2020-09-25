class StepInformation:
    """
    This steps contains information necesary for creating a pipeline and steps by the step factory

    :param step: The step
    :param pipeline: The pipeline
    """

    def __init__(self, step, pipeline):
        self.step = step
        self.pipeline = pipeline
