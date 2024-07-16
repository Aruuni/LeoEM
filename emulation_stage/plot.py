import json
import matplotlib.pyplot as plt
import glob

def extract_throughput(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    throughput = []
    for interval in data['intervals']:
        throughput.append(interval['sum']['bits_per_second'] / 1e6)  # Convert to Mbps
    return throughput

def plot_throughputs(throughputs):
    plt.figure(figsize=(12, 6))
    
    for idx, throughput in enumerate(throughputs):
        plt.plot(throughput, label=f'Flow {idx+1}')
    
    plt.xlabel('Interval')
    plt.ylabel('Throughput (Mbps)')
    plt.title('Throughput of iperf3 Flows')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Assuming all iperf JSON files are named 'iperf_c<client_number>.txt'
    files = glob.glob('iperf_c*.txt')
    
    all_throughputs = []
    for file in files:
        throughput = extract_throughput(file)
        all_throughputs.append(throughput)
    
    plot_throughputs(all_throughputs)