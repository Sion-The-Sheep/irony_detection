import json

class Parameter:
    def __init__(self) -> None:
        pass

    def set(self, *args):
        # dict
        if len(args) == 1:
            parameter = args[0]
            for key, value in parameter.items():
                setattr(self, key, value)
        # not dict
        elif len(args) == 2:
            setattr(self, args[0], args[1])
    
def init(parameterPath: str):
    with open(parameterPath, mode="r") as f:
        parameterDict = json.load(f)

    param = Parameter()
    
    for key, value in parameterDict.items():
        setattr(param, key, value)

    return param