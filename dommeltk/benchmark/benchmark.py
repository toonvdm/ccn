#! /usr/bin/env python3
import sys
import yaml
import torch

from dommel.datastructs import Dict, TensorDict
from dommel.nn import module_factory, summary

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 3:
        print("Benchmark a model based on a dommel yaml")
        print("usage: ./benchmark [summary|time] <configuration.yml> <device>")
        exit(-1)

    mode = args[1]

    config_file = args[2]
    config = Dict(yaml.load(open(config_file, "r"),
                            Loader=yaml.FullLoader))

    device = "cpu"
    if len(args) >= 4:
        device = args[3]

    inputs = TensorDict({})
    for key, shape in config.input.items():
        if shape:
            inputs[key] = torch.randn(shape)
        else:
            inputs[key] = None

    if mode == "summary":
        for key in config.keys():
            if key == "input":
                continue

            print("Summary for", key)
            model = module_factory(**config[key])
            summary(model, inputs)

    elif mode == "time":
        for key in config.keys():
            if key == "input":
                continue

            print("Time for", key, "on device", device)
            model = module_factory(**config[key])
            inputs.to(device)
            model.to(device)

            import timeit
            number = 1000
            time = timeit.timeit(stmt='outputs = model(inputs)',
                                 # already run once in setup as warmup
                                 setup='outputs = model(inputs)',
                                 number=number, globals=globals()) / number
            print(time * 1000, "ms")
