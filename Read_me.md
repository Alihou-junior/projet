# Project: Evaluating the Efficiency of Data Parallelism in Machine Learning within NEF

## Goal:  The objective of this project is to demonstrate the benefits of parallelizing the training process for machine learning models.

## Start I: Test Demo
- Book fours cores in NEF cluster
- Clone the directory `git clone https://gitlab.inria.fr/chxu/pytorch_exercice.git`
- Enter into the "project" directory
- Activate your conda environment
- Execute the code `torchrun --nnodes=1 --nproc-per-node=1 demo.py`
- Execute the code `torchrun --nnodes=1 --nproc-per-node=4 demo.py`
- Do you see the time difference?

## Start II: Test Demo_gpu
- Book two GPUs in NEF cluster
- Enter into the "project" directory
- Activate your conda environment
- Execute the code `torchrun --nnodes=1 --nproc-per-node=1 demo_gpu.py`
- Execute the code `torchrun --nnodes=1 --nproc-per-node=2 demo_gpu.py`
- Do you see the time difference?

## Requirement for the project, please check the slide!

## Supplementary details for the plots:
- Impact of the computing nodes (4 plots): 
  - cpu (two plots): 
    - time (loading time, computing and communication time) vs number of devices 
    - Throughput (batch size/time) vs number of devices
  - gpu (two plots)
    - time (loading time, computing and communication time) vs number of devices 
    - Throughput (batch size/time) vs number of devices

- Impact of the batch size (4 plots):
  - Three throughput plots vs number of gpus devices for batch size 16, 64, 128 respectively
  - Optimal number of gpus vs batch sizes