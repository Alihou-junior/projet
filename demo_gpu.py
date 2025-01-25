import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import time
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
from torch.utils.data import DataLoader
import os
from torch.nn.parallel import DistributedDataParallel as DDP


def run(rank, size):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.Imagenette('/data/neo/user/chxu/', transform=transform_train)
    dataset_size = len(dataset)
    localdataset_size = dataset_size//size
    local_dataset = torch.utils.data.Subset(dataset, range(rank*localdataset_size, (rank+1)*localdataset_size))
    sample_size = 32//size
    dataloader = DataLoader(local_dataset, batch_size=sample_size, shuffle=True)
    model = models.resnet18().to(device_id)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes)).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)


    print(f"Start running basic DDP example on rank {rank} with model Resnet18.")
    st = time.time()
    train_images, train_labels = next(iter(dataloader))
    train_images = train_images.to(device_id)
    train_labels = train_labels.to(device_id)
    et_read = time.time()
    print(f'Loading time: {et_read-st} seconds')
    optimizer.zero_grad()
    outputs = ddp_model(train_images)
    loss_fn(outputs, train_labels).backward()
    et = time.time()
    print(f'Computing + Communication time: {et-et_read} seconds')
    optimizer.step()
    dist.destroy_process_group()
    print(f"Finished running basic DDP example on rank {rank}.")

if __name__ == "__main__":
    dist.init_process_group("nccl", init_method="env://")
    size = dist.get_world_size()
    rank = dist.get_rank()
    run(rank, size)