import os
import yaml
from copy import deepcopy
from dommel.datastructs import Dict

_supported_operands = ("+", "-", "*", "/", "//")
_deref = "$"


class Loader(yaml.FullLoader):
    def __init__(self, stream):
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def construct_include(loader, node):
    filename = os.path.abspath(os.path.join(
        loader._root, loader.construct_scalar(node)))

    with open(filename, 'r') as f:
        return yaml.load(f, Loader)


yaml.add_constructor('!include', construct_include, Loader)


def parse(file):
    params = {}
    content = yaml.load(open(file, "r"), Loader)
    if "params" in content:
        params = deepcopy(content["params"])
        del content["params"]
        composites = {k: v for k, v in params.items() if isinstance(v, str)}
        constant_keys = params.keys() - composites
        composite_keys = list(composites.keys())
        constants = {k: v for k, v in params.items() if k in constant_keys}
        for k in constant_keys:
            constants[f"${k}"] = constants.pop(k)
        last_composite_keys = None
        while composite_keys:
            if last_composite_keys == composite_keys:
                print(
                    "Couldn't resolve any variables this iteration,"
                    " check for loops in your yaml"
                )
                return
            last_composite_keys = deepcopy(composite_keys)
            composites = {
                k: v for k, v in composites.items() if k in composite_keys
            }
            for k, v in composites.items():
                parsed = _parse_string(v, constants)
                if parsed:
                    constants[f"${k}"] = parsed
                    composite_keys.remove(k)
    if params:
        _descend(content, constants)
    return Dict(content)


def _parse_string(string, params):
    string = deepcopy(string)
    if isinstance(string, str):
        if any(op in string for op in _supported_operands) or any(
            p in string for p in params
        ):
            original_string = deepcopy(string)
            for key, value in params.items():
                string = string.replace(key, str(value))
            try:
                evalled = eval(string)
                return evalled
            except (NameError, SyntaxError):
                if not handle_path(string):
                    print(f"Error parsing {original_string}")
                return None
    return None


def handle_path(string):
    if "/" in string:
        print(f"Interpreting {string} as a path")
        return True
    return False


def _descend(d, params):
    for k, v in d.items():
        if isinstance(v, str):
            parsed = _parse_string(v, params)
            if parsed:
                d[k] = parsed
        elif isinstance(v, dict):
            _descend(v, params)
        elif isinstance(v, list):
            for i, el in enumerate(v):
                if isinstance(el, dict):
                    _descend(el, params)
                parsed = _parse_string(el, params)
                if parsed:
                    v[i] = parsed
        else:
            # leave the non-string and non-dicts as they are
            pass
