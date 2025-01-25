import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import time
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
import os
import csv
from torch.utils.data import DataLoader, Subset
from torch.nn.parallel import DistributedDataParallel as DDP

def get_model(model_name, num_classes, device):
    if hasattr(models, model_name):
        model_func = getattr(models, model_name)
        model = model_func(weights=None)  # Remplace pretrained=False
        if hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
        elif hasattr(model, 'classifier'):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes).to(device)
        return model.to(device)
    else:
        raise ValueError(f"Model {model_name} not found in torchvision.models.")

def save_results(rank, num_devices, batch_size, load_time, compute_time):
    filename = "performance_results_gpu.csv"
    header = ["rank", "num_devices", "batch_size", "loading_time", "compute_time"]
    data = [rank, num_devices, batch_size, load_time, compute_time]
    
    # Append data to CSV file
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(header)
        writer.writerow(data)

def run(rank, size, model_name, batch_size):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))  # Définit le bon GPU
    device_id = int(os.environ["LOCAL_RANK"])
    device = torch.device(f'cuda:{device_id}')

    # Transformations des données
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Chargement du dataset et partitionnement
    dataset = torchvision.datasets.Imagenette('/data/neo/user/chxu/', transform=transform_train)
    dataset_size = len(dataset)
    localdataset_size = dataset_size // size
    local_dataset = Subset(dataset, range(rank * localdataset_size, (rank + 1) * localdataset_size))
    local_batch_size = batch_size // size
    dataloader = DataLoader(local_dataset, batch_size=local_batch_size, shuffle=True)

    # Création du modèle et envoi sur GPU
    model = get_model(model_name, len(dataset.classes), device)
    ddp_model = DDP(model, device_ids=[device_id])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    print(f"Start running DDP on rank {rank} with model {model_name}, batch size {batch_size}.")
    
    # Chronométrage du chargement des données
    st = time.time()
    train_images, train_labels = next(iter(dataloader))
    train_images, train_labels = train_images.to(device), train_labels.to(device)
    et_read = time.time()
    load_time = et_read - st
    print(f'Loading time (rank {rank}): {load_time:.4f} seconds')

    # Chronométrage du calcul
    optimizer.zero_grad()
    outputs = ddp_model(train_images)
    loss_fn(outputs, train_labels).backward()
    et = time.time()
    compute_time = et - et_read
    print(f'Compute + Communication time (rank {rank}): {compute_time:.4f} seconds')

    optimizer.step()

    # Sauvegarde des résultats
    save_results(rank, size, batch_size, load_time, compute_time)

    dist.barrier()  # Synchronisation entre les processus
    dist.destroy_process_group()  # Fermeture propre

    print(f"Finished running DDP on rank {rank}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Nom du modèle (e.g., resnet18, vgg16, densenet121)")
    parser.add_argument("batch_size", type=int, help="Taille du batch global")
    args = parser.parse_args()

    dist.init_process_group("nccl", init_method="env://")  # Initialisation du backend
    size = dist.get_world_size()
    rank = dist.get_rank()
    run(rank, size, args.model_name, args.batch_size)
