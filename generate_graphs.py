import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Fichier de données CPU et GPU
cpu_results_file = "timing_results.txt"
gpu_results_file = "gpu_timing_results.txt"

# Chargement des fichiers de résultats
def load_results(file_path):
    """Charge les résultats à partir d'un fichier texte."""
    data = pd.read_csv(file_path, sep=", ", engine="python")
    data.columns = [col.strip() for col in data.columns]
    return data

# Calcul du débit
def calculate_throughput(data):
    """Ajoute une colonne de débit au DataFrame."""
    data["Total Time"] = data["Loading time"] + data["Computation+Communication time"]
    data["Throughput"] = data["Batch size"] / data["Total Time"]
    return data

# Générer les graphes
def plot_time_vs_devices(data, title, output_path):
    """Trace Temps vs Nombre de dispositifs."""
    devices = data["Rank"].nunique()
    mean_data = data.groupby("Rank")[["Loading time", "Computation+Communication time"]].mean()
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(devices), mean_data["Loading time"], label="Loading Time", marker="o")
    plt.plot(range(devices), mean_data["Computation+Communication time"], label="Computation+Communication Time", marker="s")
    plt.title(title)
    plt.xlabel("Number of Devices")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()

def plot_throughput_vs_devices(data, title, output_path):
    """Trace Débit vs Nombre de dispositifs."""
    devices = data["Rank"].nunique()
    mean_data = data.groupby("Rank")["Throughput"].mean()
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(devices), mean_data, label="Throughput", marker="o")
    plt.title(title)
    plt.xlabel("Number of Devices")
    plt.ylabel("Throughput (images/second)")
    plt.grid()
    plt.savefig(output_path)
    plt.close()

def plot_batch_throughput(data, batch_sizes, title_prefix, output_folder):
    """Trace le débit pour différentes tailles de batch."""
    for batch in batch_sizes:
        batch_data = data[data["Batch size"] == batch]
        devices = batch_data["Rank"].nunique()
        mean_throughput = batch_data.groupby("Rank")["Throughput"].mean()
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(devices), mean_throughput, label=f"Batch Size {batch}", marker="o")
        plt.title(f"{title_prefix} (Batch Size {batch})")
        plt.xlabel("Number of Devices")
        plt.ylabel("Throughput (images/second)")
        plt.grid()
        plt.legend()
        plt.savefig(f"{output_folder}/throughput_batch_{batch}.png")
        plt.close()

def plot_optimal_gpus(data, batch_sizes, output_path):
    """Trace le nombre optimal de GPUs pour différentes tailles de batch."""
    optimal_gpus = []
    for batch in batch_sizes:
        batch_data = data[data["Batch size"] == batch]
        mean_throughput = batch_data.groupby("Rank")["Throughput"].mean()
        optimal_gpu = mean_throughput.idxmax()
        optimal_gpus.append((batch, optimal_gpu))
    
    batch_sizes, optimal_gpus = zip(*optimal_gpus)
    
    plt.figure(figsize=(8, 5))
    plt.plot(batch_sizes, optimal_gpus, label="Optimal GPUs", marker="o")
    plt.title("Optimal Number of GPUs vs Batch Sizes")
    plt.xlabel("Batch Size")
    plt.ylabel("Optimal Number of GPUs")
    plt.grid()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    # Charger les données CPU et GPU
    cpu_data = calculate_throughput(load_results(cpu_results_file))
    gpu_data = calculate_throughput(load_results(gpu_results_file))
    
    # Graphiques d'impact des nœuds de calcul
    plot_time_vs_devices(cpu_data, "CPU: Time vs Number of Devices", "cpu_time_vs_devices.png")
    plot_throughput_vs_devices(cpu_data, "CPU: Throughput vs Number of Devices", "cpu_throughput_vs_devices.png")
    plot_time_vs_devices(gpu_data, "GPU: Time vs Number of Devices", "gpu_time_vs_devices.png")
    plot_throughput_vs_devices(gpu_data, "GPU: Throughput vs Number of Devices", "gpu_throughput_vs_devices.png")
    
    # Graphiques d'impact de la taille de batch
    batch_sizes = [16, 64, 128]
    plot_batch_throughput(gpu_data, batch_sizes, "GPU Throughput", ".")
    plot_optimal_gpus(gpu_data, batch_sizes, "optimal_gpus_vs_batch.png")

    print("Graphiques générés et enregistrés.")
