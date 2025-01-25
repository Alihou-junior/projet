import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import time
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# Function to dynamically assign GPU to a process
def prepare_device(rank):
    """
    Prepares the GPU device for the current process.
    Args:
        rank (int): Rank of the current process in the distributed setup.
    
    Returns:
        device (torch.device): The GPU device assigned to the process.
        device_ids (list): List of GPU device IDs for this process.
    """
    # Get the number of GPUs available
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for training.")

    # Dynamically assign GPU to the current process
    device_id = rank % num_gpus
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    return device, [device_id]

# Function to run distributed training
def run(rank, size, model_name, batch_size, data_path, output_file):
    """
    Runs the distributed training task for a single process.
    
    Args:
        rank (int): Rank of the current process in the distributed setup.
        size (int): Total number of processes (world size).
        model_name (str): Name of the model to be trained.
        batch_size (int): Global batch size (split across processes).
        data_path (str): Path to the dataset.
        output_file (str): File to save timing results.
    """
    # Prepare the device and assign a GPU
    device, device_ids = prepare_device(rank)
    
    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load dataset and split for distributed training
    dataset = torchvision.datasets.Imagenette(data_path, transform=transform_train)
    dataset_size = len(dataset)
    local_dataset_size = dataset_size // size
    local_dataset = torch.utils.data.Subset(dataset, range(rank * local_dataset_size, (rank + 1) * local_dataset_size))
    sample_size = batch_size // size
    dataloader = DataLoader(local_dataset, batch_size=sample_size, shuffle=True)
    
    # Create and prepare the model
    model = models.__dict__[model_name]().to(device)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes)).to(device)
    ddp_model = DDP(model, device_ids=device_ids)
    
    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    # Print initialization
    print(f"Start running distributed training on rank {rank} with model {model_name}.")
    
    # Measure loading time
    st = time.time()
    train_images, train_labels = next(iter(dataloader))
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    et_read = time.time()
    loading_time = et_read - st
    
    # Measure computation + communication time
    optimizer.zero_grad()
    outputs = ddp_model(train_images)
    loss = loss_fn(outputs, train_labels)
    loss.backward()
    et_compute = time.time()
    computation_time = et_compute - et_read
    optimizer.step()
    
    # Log results to file
    if rank == 0:  # Only process rank 0 writes results
        with open(output_file, "a") as f:
            f.write(f"Model: {model_name}, Batch size: {batch_size}, Rank: {rank}, Loading time: {loading_time:.4f}s, "
                    f"Computation + Communication time: {computation_time:.4f}s\n")
    
    # Destroy the process group
    dist.destroy_process_group()
    print(f"Finished training on rank {rank}.")

# Main block
if __name__ == "__main__":
    # Initialize the distributed process group
    dist.init_process_group("nccl", init_method="env://")
    size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Define parameters
    model_name = "resnet18"  # Replaceable with any supported model
    batch_sizes = [16, 32, 64, 128]
    data_path = "/data/neo/user/chxu/"
    output_file = "gpu_timing_results.txt"
    
    # Clear the output file if rank 0
    if rank == 0 and os.path.exists(output_file):
        open(output_file, "w").close()
    
    # Run the training process 5 times for each batch size
    for batch_size in batch_sizes:
            for _ in range(5):  # Effectue 5 tests pour chaque cas
                run(rank, size, model_name, batch_size, data_path, output_file)
