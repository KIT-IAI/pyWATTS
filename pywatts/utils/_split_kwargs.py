def split_kwargs(kwargs):
    inputs = {}
    targets = {}
    for key, value in kwargs.items():
        if key.startswith("target"):
            targets[key] = value
        else:
            inputs[key] = value
    return inputs, targets
