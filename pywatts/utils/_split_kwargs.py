
def split_kwargs(kwargs):
    inputs = dict()
    targets = dict()
    for key, value in kwargs.items():
        if key.startswith("target"):
            targets[key] = value
        else:
            inputs[key] = value
    return  inputs, targets