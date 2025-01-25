import pandas as pd
import matplotlib.pyplot as plt

def plot_time_vs_devices(filename):
    df = pd.read_csv(filename)
    grouped = df.groupby("num_devices").mean()
    
    plt.figure(figsize=(10, 5))
    plt.plot(grouped.index, grouped["loading_time"], label="Loading Time", marker='o')
    plt.plot(grouped.index, grouped["compute_time"], label="Computing + Communication Time", marker='s')
    plt.xlabel("Number of Devices")
    plt.ylabel("Time (seconds)")
    plt.title("Time vs Number of Devices (GPU)")
    plt.legend()
    plt.grid()
    plt.savefig("time_vs_devices_gpu.png")
    plt.show()

def plot_throughput_vs_devices(filename):
    df = pd.read_csv(filename)
    df["throughput"] = df["batch_size"] / (df["loading_time"] + df["compute_time"])
    grouped = df.groupby("num_devices").mean()
    
    plt.figure(figsize=(10, 5))
    plt.plot(grouped.index, grouped["throughput"], label="Throughput", marker='o', color='red')
    plt.xlabel("Number of Devices")
    plt.ylabel("Throughput (images per second)")
    plt.title("Throughput vs Number of Devices (GPU)")
    plt.legend()
    plt.grid()
    plt.savefig("throughput_vs_devices_gpu.png")
    plt.show()

if __name__ == "__main__":
    data_file = "performance_results_gpu.csv"
    plot_time_vs_devices(data_file)
    plot_throughput_vs_devices(data_file)
