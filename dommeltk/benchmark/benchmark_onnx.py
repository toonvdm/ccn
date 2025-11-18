#! /usr/bin/env python3
import sys
import numpy as np


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 3:
        print("Benchmark a model from a onnx file")
        print("usage: ./benchmark [summary|time] <model.onnx>")
        exit(-1)

    mode = args[1]
    onnx_file = args[2]

    if mode == "summary":
        import onnx
        model = onnx.load(onnx_file)
        print(onnx.helper.printable_graph(model.graph))

    elif mode == "time":
        import onnxruntime as ort
        import timeit

        ort_session = ort.InferenceSession("graspnet.onnx")
        nn_inputs = {}
        for nn_input in ort_session.get_inputs():
            nn_inputs[nn_input.name] = np.random.randn(
                *nn_input.shape).astype(np.float32)

        number = 1000
        time = timeit.timeit(
            stmt='nn_outputs = ort_session.run(None, nn_inputs)',
            # already run once in setup as warmup
            setup='nn_outputs = ort_session.run(None, nn_inputs)',
            number=number, globals=globals()) / number
        print("Time", time * 1000, "ms")
