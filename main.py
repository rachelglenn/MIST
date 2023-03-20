import json
import os

from analyze_data.analyze import Analyze
from preprocess_data.preprocess import preprocess_dataset
from runtime.args import get_main_args
from runtime.run import RunTime
from runtime.utils import create_empty_dir, set_warning_levels


def create_folders(args):
    data_path = os.path.abspath(args.data)
    with open(data_path, "r") as file:
        data = json.load(file)

    is_test = False
    if "test-data" in data.keys():
        is_test = True

    results = os.path.abspath(args.results)
    processed_data = os.path.abspath(args.processed_data)

    dirs_to_create = [results,
                      os.path.join(results, "predictions"),
                      os.path.join(results, "predictions", "train"),
                      os.path.join(results, "predictions", "train", "raw"),
                      os.path.join(results, "predictions", "train", "postprocess"),
                      os.path.join(results, "predictions", "train", "postprocess", "clean_mask"),
                      os.path.join(results, "predictions", "train", "final"),
                      os.path.join(results, "models"),
                      os.path.join(results, "models", "best"),
                      os.path.join(results, "models", "last")]

    labels = data["labels"]
    for i in range(1, len(labels)):
        dirs_to_create.append(os.path.join(results, "predictions", "train", "postprocess", str(labels[i])))

    if is_test:
        dirs_to_create.append(os.path.join(results, "predictions", "test"))

    for folder in dirs_to_create:
        create_empty_dir(folder)

    create_empty_dir(processed_data)


def main(args):

    set_warning_levels()

    # Create file structure for MIST output
    create_folders(args)

    if args.exec_mode == "all":
        analyze = Analyze(args)
        analyze.run()

        preprocess_dataset(args)
        print("completed preprocessing")
        runtime = RunTime(args)
        runtime.run()

    elif args.exec_mode == "analyze":
        analyze = Analyze(args)
        analyze.run()

    elif args.exec_mode == "preprocess":
        preprocess_dataset(args)

    elif args.exec_mode == "train":
        runtime = RunTime(args)
        runtime.run()


if __name__ == "__main__":
    args = get_main_args()

    aPath = '--xla_gpu_cuda_data_dir=/rsrch1/ip/rglenn1/support_packages/miniconda/conda_gpu4/pkgs/cuda-nvcc-11.7.64-0'
    #print(aPath)
    os.environ['XLA_FLAGS'] = aPath
    main(args)
    
    

# python main.py --exec-mode train --data /rsrch1/ip/rglenn1/data/subset_VF_dataset/dataset.json --processed-data /rsrch1/ip/rglenn1/data/subset_VF_dataset/numpy --results /rsrch1/ip/rglenn1/data/subset_VF_dataset/results --model nnunet --pocket --xla --amp --patch-size 256 256 32 --epochs 125


#export XLA_FLAGS='--xla_gpu_cuda_data_dir=/rsrch1/ip/rglenn1/support_packages/miniconda/conda_gpu4/pkgs/cuda-nvcc-11.7.64-0'


#args.data = '/rsrch1/ip/rglenn1/data/subset_VF_dataset/dataset.json'
#args.results = '/rsrch1/ip/rglenn1/data/subset_VF_dataset/results'
#args.model='uunet'
#args.pocket = True
#args.xla = True
#args.amp = True
#args.patch_size=[256, 256, 32]
#args.epochs = 125
#args.exec_mode = True
#args.train = True

# Fix not finding the gpu
#conda uninstall protobuf
#
#conda uninstall tensorflow
# python 3.10, nvcc 11.6

"""
conda install -c nvidia/label/cuda-11.6.0 cuda-nvcc
nvcc --version
export CUDA_VISIBLE_DEVICES=0,1

conda install -c "nvidia/label/cuda-11.6.0" cuda-runtime


conda install -c conda-forge tensorflow==2.10
python
import tensorflow as tf
tf.config.list_physical_devices()

#[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:6', device_type='GPU')]

export C_INCLUDE_PATH=/rsrch1/ip/rglenn1/support_packages/miniconda/conda_gpu4/pkgs/libprotobuf-3.21.12-h3eb15da_0/include/::/rsrch1/ip/rglenn1/support_packages/miniconda/conda_gpu4/pkgs/cuda-nvcc-11.6.55-h5758ece_0/include/:/rsrch1/ip/rglenn1/support_packages/miniconda/conda_gpu4/pkgs/cuda-cudart-11.3.58-hc1aae59_0/include/:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/rsrch1/ip/rglenn1/support_packages/miniconda/conda_gpu4/pkgs/libprotobuf-3.21.12-h3eb15da_0/include/::/rsrch1/ip/rglenn1/support_packages/miniconda/conda_gpu4/pkgs/cuda-nvcc-11.6.55-h5758ece_0/include/:/rsrch1/ip/rglenn1/support_packages/miniconda/conda_gpu4/pkgs/cuda-cudart-11.3.58-hc1aae59_0/include/:$CPLUS_INCLUDE_PATH


pip3 install --upgrade --force-reinstall   --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
pip3 install --upgrade --force-reinstall  --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-tf-plugin-cuda110



"""

