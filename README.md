# Testing Gradient Platform 
Repo to test gradient notebooks and verify if all the libraries installed properly. And to ensure proper GPU utilisation

### Tests to run to ensure Environment setup correctly
- Basic Sanity tests these should work out of the box!
```
# check if nvidia driver and toolkit is installed correctly
nvidia-smi

# check if LD_LIBRARY_PATH is setup properly
python -m bitsandbytes

# quick check on TF being able to access GPUs
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

- `CUDNN` tests:
```
# Run cudnn sample - mnistCUDNN
cd cudnn_samples_v8/mnistCUDNN
make clean
make

# Read this if you get into errors:https://forums.developer.nvidia.com/t/freeimage-is-not-set-up-correctly-please-ensure-freeimae-is-set-up-correctly/66950/3
sudo apt-get install libfreeimage3 libfreeimage-dev

make clean
make

./mnistCUDNN  # this should run without any tests failing.
```

- `Pytorch` GPU tests:
```
cd nbs

# open pytorch.ipynb
# run cells
# observe that GPU(s) detected and being utilised in metrics / nvidia-smi

# similarly run cells in pytorch-multi-gpu.ipynb if its a multi gpu notebook!
```

- `Tensorflow` GPU tests:
```
cd nbs

# open tf-begineer.ipynb
# run cells
# observe that GPU(s) detected and being utilised in metrics / nvidia-smi

# similarly run cells in tf-advanced.ipynb
```

- `Accelerate` [Single GPU test and Multi-GPU test](accelerate_nlp_test/README.md)

### Example Test run results
- Everything works as expected, need to see if we get approval for non deb based installation.

| GPU/Family       | Nvidia-smi?      | PyTorch?     | Tensorflow?    | CudaNN test? | 
| -------------    | -------------    | -------------| -------------  | -------------|
| P6000            | Yes              | Yes          | Yes            | Yes          |
| RTX4000          | Yes              | Yes          | Yes            | Yes          |
| A6000            | Yes              | Yes          | Yes            | Yes          |
| V100-32          | Yes              | Yes          | Yes            | Yes          |
| A100             | Yes              | Yes          | Yes            | Yes          |


- Accelerate/Multi GPU tests:

| GPU/Family       | Nvidia-smi?      | Accelerate?  | PyTorch-MultiGPU? | 
| -------------    | -------------    | -------------|-------------|
| A100             | Yes              | Yes          | NA          |
| A4000x2          | Yes              | Yes          | Yes         |

- Note: The actual machine instance might change, the key is to test one from each GPU Arch family.