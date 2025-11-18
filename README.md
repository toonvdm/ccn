# Object-Centric Scene Representations using Active Inference 

This repository contains the code for the experiments used in [Object-Centric Scene Representations using Active Inference](https://arxiv.org/abs/2302.03288) by Toon Van de Maele, Tim Verbelen, Pietro Mazzaglia, Stefano Ferraro, and Bart Dhoedt. 

## Code structure

- `scene-environments`: contains the code for the environments
- The code for the model and the active inference agent is in `ccn` 
- `dommeltk` is a framework we built around pytorch to aid us in development and is required to run the `ccn` models. 

## Download data assets & trained models

The trained models/datasets/evaluation scenes can be downloaded in the following [link](https://rb.gy/mcl0v6). 

To run the benchmark, please move the downloaded files into the following directories: 
```
- ccn-data -> ccn/data
- scene-environments-data -> scene-environments/data 
```

## Citation 

If you find the code useful, please refer to our work using:

```
@misc{vandemaele2023objectcentricscenerepresentationsusing,
      title={Object-Centric Scene Representations using Active Inference}, 
      author={Toon Van de Maele and Tim Verbelen and Pietro Mazzaglia and Stefano Ferraro and Bart Dhoedt},
      year={2023},
      eprint={2302.03288},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2302.03288}, 
}
```