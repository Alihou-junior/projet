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

# Function to initialize the dataset and DataLoader
#size = nbre total de processus used dans l'entrainement(taille du cluster)
#rank : id unique d'un processus dans la distrib
#batch_size : taille du batch (qui sera divisé entre les process)
#data_path : dataset(chemin vers données)
def prepare_dataloader(rank, size, batch_size, data_path):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = torchvision.datasets.Imagenette(data_path, transform=transform_train)#chargement du dataset avec les transformation de transform_train
    dataset_size = len(dataset)
    local_dataset_size = dataset_size // size #Le dataset est divisé pour chaque processus selon le rank afin que chaque nœud ne traite qu'une partie des données.
    local_dataset = torch.utils.data.Subset(dataset, range(rank * local_dataset_size, (rank + 1) * local_dataset_size))

    dataloader = DataLoader(local_dataset, batch_size=batch_size // size, shuffle=True) #loader avec des ous données , ajustement de la taille du batch
    return dataloader, len(dataset.classes)

# Function to create a model dynamically
def create_model(model_name, num_classes):
    # Dynamically get the model from torchvision.models
    model_class = getattr(models, model_name, None)#getattr permet d'accéder dynamiquement à une méthode/classe au sein de torchvision.models
    if model_class is None:
        raise ValueError(f"Model {model_name} is not supported in torchvision.models.")
    
    # Instantier le model
    model = model_class()
    
    # Modify the last fully connected layer to match the number of classes
    if hasattr(model, "fc"):  #check si le le model a une couche fully connected
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):  # pour les models qui utilisent classifier comme MobileNet 
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        else:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unable to modify the output layer for model {model_name}.")
    
    return model


# Main function to run the training
def run(rank, size, model_name, batch_size, data_path, output_file):
    # Prepare DataLoader
    dataloader, num_classes = prepare_dataloader(rank, size, batch_size, data_path)

    # Create model and wrap it in DDP
    model = create_model(model_name, num_classes)
    ddp_model = DDP(model)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    print(f"Start running DDP training on rank {rank} with model {model_name}.")

    # Measure loading time
    print("debut de la mesure du load")
    st = time.time()
    train_images, train_labels = next(iter(dataloader))
    et_read = time.time()
    loading_time = et_read - st

    print("fin de la mesure du load")

    print("debut de la mesure du comput + communication")

    # Measure computation + communication time
    optimizer.zero_grad()
    outputs = ddp_model(train_images)
    loss = loss_fn(outputs, train_labels)
    loss.backward()
    et_compute = time.time()
    computation_time = et_compute - et_read
    print("fin de la mesure du comput + communication")

    print("optimisation")

    optimizer.step()

    # Save times to a file
    if rank == 0:  # Only write to file from rank 0(leader)
        with open(output_file, "a") as f:
            f.write(f"Model: {model_name}, Batch size: {batch_size}, Rank: {rank}, "
                    f"Loading time: {loading_time:.4f}s, Computation+Communication time: {computation_time:.4f}s\n")

    dist.destroy_process_group()#liberer le groupe de processus utilisé
    print(f"Finished running DDP training on rank {rank}.")

if __name__ == "__main__":
    # Initialize distributed training
    print(f"Initializing process group on rank {os.environ.get('RANK', 'undefined')}...")
    dist.init_process_group("gloo", init_method="env://")
    size = dist.get_world_size()
    rank = dist.get_rank()
    print(f"Process group initialized: rank {rank}/{size}")
    print("fin de l'initialisation")

    # Parameters (can be modified dynamically)
    model_name = "resnet34"  # Change to others as needed
    batch_size = 16  # Change to 16, 64, 128 for tests
    data_path = "/data/neo/user/chxu/"  # Update to your dataset path
    output_file = "timing_results.txt"  # File to save the timing results

    print("ouverture et ecriture dans le fichier")
    # Ensure the output file is cleared before writing new results (only on rank 0)
    if rank == 0 and os.path.exists(output_file):
        open(output_file, "w").close()

    print("fin fichier")
    # Run the training process 5 times for each batch size
    for _ in range(5):  # Effectue 5 tests pour chaque cas
        run(rank, size, model_name, batch_size, data_path, output_file)
