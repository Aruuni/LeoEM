import json
import matplotlib.pyplot as plt

def plot_iperf_data(json_file):
    # Load the iperf3 JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # iperf3 results are in data["intervals"], each containing:
    #   "sum": { ... }        (aggregated data across all streams)
    #   "streams": [ { ... } ] (detailed data per stream)
    intervals = data["intervals"]
    
    # We will collect the following metrics at each interval:
    #   1) Time (seconds) – we can take the "end" field from "sum" as the interval's timestamp
    #   2) Sum Throughput (Mbps) – from "sum.bits_per_second"
    #   3) Sum Retransmissions – from "sum.retransmits"
    #   4) (Optionally) one stream's RTT (µs) – from "streams"[0]["rtt"]
    #   5) (Optionally) one stream's congestion window (bytes) – from "streams"[0]["snd_cwnd"]
    
    time_vals = []
    sum_throughput = []
    sum_retrans = []
    
    # If you have guaranteed exactly one stream, you can just pick stream[0].
    # Otherwise, you might loop over streams or choose a different approach.
    stream_rtt = []
    stream_cwnd = []
    
    for interval in intervals:
        sum_data = interval["sum"]
        stream_data = interval["streams"][0]  # first (or only) stream in this interval
        
        # Use interval's "end" field from the sum to represent the time axis
        time_vals.append(sum_data["end"])
        
        # Convert bits_per_second to Mbps
        sum_throughput.append(sum_data["bits_per_second"] / 1e6)
        
        # Sum retransmissions
        sum_retrans.append(sum_data["retransmits"])
        
        # Stream's RTT (microseconds)
        # iperf3 typically stores RTT in microseconds
        stream_rtt.append(stream_data["rtt"])
        
        # Stream's congestion window (bytes)
        stream_cwnd.append(stream_data["snd_cwnd"])
    
    # Plot everything in one figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1) Sum Throughput
    axs[0, 0].plot(time_vals, sum_throughput, color='b',  label='Sum Throughput (Mbps)')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Throughput (Mbps)')
    axs[0, 0].set_title('Sum Throughput over Time')
    axs[0, 0].grid(True)
    
    # 2) Sum Retransmissions
    axs[0, 1].plot(time_vals, sum_retrans, color='r',  label='Sum Retransmits')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Retransmissions')
    axs[0, 1].set_title('Sum Retransmissions over Time')
    axs[0, 1].grid(True)
    
    # 3) Stream RTT
    axs[1, 0].plot(time_vals, stream_rtt, color='g',  label='Stream RTT (µs)')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('RTT (µs)')
    axs[1, 0].set_title('Stream RTT over Time')
    axs[1, 0].grid(True)
    
    # 4) Stream CWND
    axs[1, 1].plot(time_vals, stream_cwnd, color='m',  label='Stream CWND (bytes)')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Congestion Window (bytes)')
    axs[1, 1].set_title('Stream Congestion Window over Time')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Change this to your actual JSON file path.
    json_file = "/home/mihai/Desktop/LeoEM/emulation_stage/Starlink_NY_LDN_15_ISL_path/QSize_5000/2_flows/algo_bbr/run_1/iperf_c1.json"
    plot_iperf_data(json_file)
