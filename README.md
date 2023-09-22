# ML approaches for SuperNEMO tracking
For results see SuperNEMO private documentation. If you have any question feel free to contact me [adam.mendl@cvut.cz](mailto:adam.mendl@cvut.cz) or [amendl@hotmail.com](mailto:amendl@hotmail.com)

---

# Installation and required software
Information in this section are mostly for CCLyon in2p3 computation cluster.
## Required software
Almost everything runs ot top of `python3`. On CCLyon use `python` sourced with `root` via
1. `ccenv root 6.22.06` - loads `python 3.8.6` (**does not work now**)
2. since July 12 2023 `module add Analysis/root/6.22.06-fix01` - loads `python 3.9.1` (**currently, this is the way to go**)

This software should be installed in python or Anaconda environment (python environment is prefered since it can access  both sourced root package and all gpu related software directly, however it is still possible to make it work with Anaconda)
 * `root` - Root is not needed to be explicitly installed in python or Anaconda environment, any sourced Root on CCLyon should work - minimum tested verion 6.22.06 (since July 12 2023 6.22.06-fix01 on CCLyon). **PyROOT is required.**
 * `cudatoolkit`, `cudnn` - Should be already installed on CCLyon 
 * `tensorflow` - (ideally 2.13, older version produce some random bug with invalid version of openSSL build on CCLyon. However, there seems to be [BUG](https://github.com/tensorflow/tensorflow/issues/61314) in 2.13 regarding training on multiple GPUs)
 * `keras` - Should be part of `tensorflow`
 * `keras-tuner` - hyperparameter tuning
 * `numpy`
 * `maplotlib`, `seaborn` - plotting
 * `scikit-learn` - some helper functions
 * `pydot`, `graphviz` - drawing models
 * `argparse`
 
Optional:
 * `tensorrt` - really useful, makes inference and training of models faster on CPU. It can be also installed in two steps, first install `nvidia-pyindex` and then `nvidia-tensorrt`. **To use it succesfully within scripts, you should import it before tensorflow!**
 * `tensorboard` - Should be part of `tensorflow`
 * `tensorboard_plugin_profile` - profiling
 * `nvidia-pyindex`, `nvidia-tensorrt` - For TensorRT support
 * `nvidia-smi` -  For checking usage and available memory on NVIDIA V100 GPU (on CCLyon)
 * `ray` - For massively parallelized hyperparameter optimization. Use this command `python -m pip install -U "ray[data,train,tune,serve]"` to install.

## Running scripts (on CCLyon in2p3 cluster)
Example is at `example_exec.sh`. Run it with `sbatch --mem=... -n 1 -t ... gres=gpu:v100:N example_exec.sh` if you have access to GPU, where `N` is number of GPUs you want to use (currently CCLyon does not allow me to use more than three of them) Otherwise, leave out `gres` option.

Scripts can use two strategies. To use only one GPU use option `--OneDeviceStrategy "/gpu:0"`. If you want to use more GPUs, use for example `--MirroredStrategy "/gpu:0" "/gpu:1" "/gpu:2"`. For some reason, I was never able to use more than 3 GPUs on CClyon.

If you start job from bash instance with some packages, modules or virtual environment loaded, you should unload them/deactivate them (use `module purge --force`). Best way is to start from fresh bash instance (on many places I use environment variables to pass arguments to scripts so starting things fresh is really really important!)
## Workflow overview
1. source `root` (and `python`) - **currently use `module add Analysis/root/6.22.06-fix01`**
2. create python virtual environment (if not done yet) 
3. install [packages](#required-software) (if not done yet)
4. load python virtual environment (or add `#! <path-to-your-python-bin-in-envorinment>` to first line of your script)
## Working with real data (temporary solution)
We test models on real data and compared them with [TKEvent](https://github.com/TomasKrizak/TKEvent). Unfortunately, it is not possible to open `root` files produced by [TKEvent](https://github.com/TomasKrizak/TKEvent) library from python with the same library sourced since this library might be built with different version of python and libstdc++. Fortunately, workaround exists. We need to download and build two versions of [TKEvent](https://github.com/TomasKrizak/TKEvent). First version will be built in the manner described in [TKEvent](https://github.com/TomasKrizak/TKEvent) README.md. The second library shoudl be build (we ignore the `red_to_tk` target) with following steps:

1. `module add ROOT` where `ROOT` is version of `root` library used by `tensorflow` (**currently `module add Analysis/root/6.22.06-fix01`**)
2. `TKEvent/TKEvent/install.sh` to build library 

Now, we can use `red_to_tk` from the first library to obtain root file with `TKEvent` objects and open this root file with the second version of `TKEvent` library.
## Working with real data (future-proof)
If the collaboration will want to use keras models inside software, the best way is probably to use [cppflow](https://github.com/serizba/cppflow) . It is single header c++ library for acessing TensoFlow C api. This means that we will not have to build TensorFlow from source and we should not be restricted by root/python/gcc/libstdc++ version nor calling conventions. 
## Local modules
Not to be restricted by the organization structure of our folders, we use this script to load, register and return local modules.
```
def import_arbitrary_module(module_name,path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name,path)
    imported_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = imported_module
    spec.loader.exec_module(imported_module)

    return imported_module
```
## Issues
 1. `sbatch` and `tensorflow` sometimes fail to initialize libraries (mainly to source python from virtual environment or root) - start the script again ideally from new bash instance without any modules nor virtual environment loaded.
 2. `tensorflow` sometimes runs out of memory - Don't use checkpoints for `tensorboard`. Another cause of this problem might be training more models in one process, we can solve this by `keras.backend.clear_session()`. If this error occurs after several hours of program execution, check out function `tf.config.experimental.set_memory_growth`. 
 3. TensorFlow 2.13 distributed training fail - https://github.com/tensorflow/tensorflow/issues/61314
 4. Sometimes, there are strange errors regarding ranks of tensors while using custom training loop in `gan.py` - looks like really old still unresolved bug inside core tensorflow library. However, the workaround is to pass only one channel into CNN architecture and concat them with `tf.keras.layers.Concatenate`
 5. `numpy` ocasionally stops working (maybe connected to [this](https://github.com/pypa/pip/issues/9542) issue), failing to import (`numpy.core._multiarray_umath` or other parts of `numpy` library). Is is accompanied by message:
 ```
IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
 ``` 
 Simplest solution is to create new environment and install all libraries inside that.


## Paths
Since I use absolute paths on many places you have to update paths after downloading this project. Rule of thumb is that change is required for:
1. scripts that load data into tf.dataset pipelines (search for `TFile` and update paths to your simulations/measured data),
2. all *.sh files. Paths that require changing are: a) root, python, g++ and other libraries; b) python environments and c) scripts

## Code description
Please note, that this description is rather short. For more detailed overview, see SuperNEMO documentation. For my implementation of CapsNET, see [this repository](https://github.com/amendl/Capsules).