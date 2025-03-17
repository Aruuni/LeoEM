#!/usr/bin/env python3
from mininet.topo import Topo
from mininet.node import CPULimitedHost
from mininet.node import RemoteController
from mininet.link import TCLink
from mininet.link import TCIntf
from mininet.net import Mininet
from mininet.log import lg
from mininet.util import dumpNodeConnections
import time
import networkx as nx
import numpy
import random
import scipy.io as scio
import math
from mininet.cli import CLI
from ast import literal_eval
import sys
import socket
import subprocess
import concurrent.futures
import threading
import logging
import logging.handlers
import os
import glob
import shutil
import json
import csv

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '../..')
sys.path.append(mymodule_dir)

from core.topologies import *
from core.analysis import *
from core.utils import *
from core.emulation import *
from core.config import *

my_logger = logging.getLogger('MyLogger')
my_logger.setLevel(logging.DEBUG)
handler = logging.handlers.SysLogHandler(address='/dev/log')
my_logger.addHandler(handler)

REMOTE_CONTROLLER_IP = "127.0.0.1"
kernel_output = open('/dev/kmsg', 'w')
status_report_delayed_shift = -0.5

if len(sys.argv) != 8:
    print(f"6 args:  path_info_file, start_times, bw, queue_size, protocol, run, duration {len(sys.argv)}")
    exit()

intermediate_hop_num = 0
frame_length = 1

path_info_file = sys.argv[1]
start_times = list(map(float, sys.argv[2].strip('[]').split(',')))
bent_pipe_link_bandwidth = int(sys.argv[3])
print(bent_pipe_link_bandwidth)
unitialized_bent_pipe_delay = '0.01ms'
switch_queue_size = int(sys.argv[4])
protocol = sys.argv[5]
run = int(sys.argv[6])
duration_total = int(sys.argv[7])

NODES = len(start_times)

last_routing_path = ["NULL"]
current_routing_path = ["NULL"]
last_node_delay = ["NULL"]
current_node_delay = ["NULL"]

switches = []

def simulate_dish_gateway_sat_handover(net_dish_sat_delay):
    try:
        net = net_dish_sat_delay[0]
        dish = net_dish_sat_delay[1]
        sat = net_dish_sat_delay[2]
        delay = net_dish_sat_delay[3]
        subprocess.run(["echo", "UNIX TIME: %s: end handover triggered!" % str(time.time())], stdout=kernel_output)
        net.configLinkStatus(dish, sat, 'down')
        time.sleep(delay)
        net.configLinkStatus(dish, sat, 'up')
    except Exception as e:
        print("failed to simulate end handover:", e)

def simulate_link_break(net_and_node_i):
    try:
        net = net_and_node_i[0]
        i = net_and_node_i[1]
        subprocess.run(["echo", "UNIX TIME: %s: intermediate handover triggered!" % str(time.time())], stdout=kernel_output)
        net.configLinkStatus('s%s' % (i), 's%s' % (i + 1), 'down')
        net.configLinkStatus('s%s' % (i), 's%s' % (i + 1), 'up')
    except Exception as e:
        print("failed to simulate intermediate handover:", e)

def report_handover_status_asynchronously(handover_status, delay):
    t = threading.Thread(target=report_handover_status, args=(handover_status, delay), daemon=True)
    t.start()
    return t

def report_handover_status(handover_status, delay):
    time.sleep(delay)
    msg_from_client = str(handover_status)
    bytes_to_send = str.encode(msg_from_client)
    server_addr = ("127.0.0.1", 20001)
    udp_client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    udp_client_socket.sendto(bytes_to_send, server_addr)
    udp_client_socket.close()

def compute_link_delta(old_link, new_link):
    old_intermediate = old_link[1:-1]
    new_intermediate = new_link[1:-1]
    handover_arr = [0] * (len(new_intermediate) - 1)
    j = 0
    for i in range(0, len(new_intermediate)):
        if new_intermediate[i] in old_intermediate:
            if old_intermediate.index(new_intermediate[i]) != j:
                handover_arr[i - 1] = 1
                j = old_intermediate.index(new_intermediate[i])
            j += 1
        else:
            if i > 0 and new_intermediate[i - 1] in old_intermediate:
                handover_arr[i - 1] = 1
    return [0] + handover_arr + [0]

class MyTopo(Topo):
    def __init__(self):
        global switches
        Topo.__init__(self)
        clients = []
        servers = []
        for i in range(NODES):
            clients.append(self.addHost(f'c{i+1}', ip=f"10.0.{i+1}.101/24"))
            servers.append(self.addHost(f'x{i+1}', ip=f"10.0.{i+1}.102/24"))
        for i in range(intermediate_hop_num):
            switches.append(self.addSwitch('s%s' % i))

        for i in range(intermediate_hop_num - 1):
            self.addLink('s%s' % i, 's%s' % (i + 1), bw=bent_pipe_link_bandwidth, delay=unitialized_bent_pipe_delay, max_queue_size=switch_queue_size)

        for i in range(NODES):
            self.addLink(f"c{i+1}", "s0", bw=bent_pipe_link_bandwidth,
                         delay=unitialized_bent_pipe_delay, max_queue_size=switch_queue_size)
            self.addLink(f"x{i+1}", "s%s" % (intermediate_hop_num - 1),
                         bw=bent_pipe_link_bandwidth, delay=unitialized_bent_pipe_delay, max_queue_size=switch_queue_size)

def set_link_properties(net, node1, node2, bw, delay, max_queue_size=switch_queue_size):
    hop_a = net.getNodeByName(node1)
    hop_b = net.getNodeByName(node2)
    interfaces = hop_a.connectionsTo(hop_b)
    src_intf = interfaces[0][0]
    dst_intf = interfaces[0][1]
    src_intf.config(bw=bw, delay=delay, max_queue_size=max_queue_size, smooth_change=True)
    dst_intf.config(bw=bw, delay=delay, max_queue_size=max_queue_size, smooth_change=True)

def initialize_link(net):
    for i in range(intermediate_hop_num - 1):
        set_link_properties(net, f"s{str(i)}", f"s{str(i + 1)}", bent_pipe_link_bandwidth, unitialized_bent_pipe_delay, max_queue_size=switch_queue_size)

    for i in range(NODES):
        set_link_properties(net, f"c{i+1}", "s0", bent_pipe_link_bandwidth,
                            unitialized_bent_pipe_delay, max_queue_size=switch_queue_size)
        set_link_properties(net, f"x{i+1}", f"s{intermediate_hop_num - 1}", bent_pipe_link_bandwidth,
                            unitialized_bent_pipe_delay, max_queue_size=switch_queue_size)

def update_precomputed_link(link_info_all_cycles, net):
    print("cycle numbers:", len(link_info_all_cycles))
    global current_routing_path, last_routing_path, last_node_delay, current_node_delay
    data_path = f'{LEOEM_INSTALL_FOLDER}/satcp/report_timing_error/comparsion_1_2/latency_deltas.mat'
    data = scio.loadmat(data_path)
    latency_deltas = data['latency_deltas']
    latency_deltas = latency_deltas[0]
    signness = [1, -1]
    prediction_delays = []
    for i in range(len(link_info_all_cycles)):
        prediction_delays.append(random.choice(latency_deltas) * random.choice(signness) + status_report_delayed_shift)

    handovers = [0]
    for i in range(1, len(link_info_all_cycles)):
        last_cycle = link_info_all_cycles[i - 1]
        current_cycle = link_info_all_cycles[i]
        if current_cycle[1] == "NULL":
            handovers.append(0)
            continue
        if last_cycle[1] == "NULL":
            handovers.append(1)
            continue
        current_routing_path = current_cycle[1]
        last_routing_path = last_cycle[1]
        if current_routing_path[1] != last_routing_path[1] or current_routing_path[-2] != last_routing_path[-2]:
            handovers.append(1)
            continue
        if sum(compute_link_delta(last_routing_path, current_routing_path)) != 0:
            handovers.append(1)
            continue
        handovers.append(0)
    for index in range(len(link_info_all_cycles)):
        tstart = time.time()
        cycle = link_info_all_cycles[index][0]
        print("cycle:", index)
        if index + 1 != cycle:
            print("not reading the correct cycle!")
            return

        if index + 1 < len(link_info_all_cycles) and -1 < prediction_delays[index + 1] < 0 and handovers[index + 1] == 1:
            subprocess.run(["echo", "earlier handover report triggered"], stdout=kernel_output)
            print("handover report for cycle %s with %s delay" % (str(index + 1), str(prediction_delays[index + 1])))
            report_handover_status_asynchronously(1, 1 - -prediction_delays[index + 1])
        if index + 2 < len(link_info_all_cycles) and -2 < prediction_delays[index + 2] < -1 and handovers[index + 2] == 1:
            subprocess.run(["echo", "earlier handover report triggered"], stdout=kernel_output)
            print("handover report for cycle %s with %s delay" % (str(index + 2), str(prediction_delays[index + 2])))
            report_handover_status_asynchronously(1, 2 - -prediction_delays[index + 2])

        if index + 3 < len(link_info_all_cycles) and -3 < prediction_delays[index + 3] < -2 and handovers[index + 3] == 1:
            subprocess.run(["echo", "earlier handover report triggered"], stdout=kernel_output)
            print("handover report for cycle %s with %s delay" % (str(index + 3), str(prediction_delays[index + 3])))
            report_handover_status_asynchronously(1, 3 - -prediction_delays[index + 3])

        if link_info_all_cycles[index][1] == "NULL":
            current_routing_path = ["NULL"]
            current_node_delay = ["NULL"]
            net.configLinkStatus('s0', 's1', 'down')
            print("turn off the link due to lack of route")
            time.sleep(frame_length)
            continue
        
        new_routing_path = link_info_all_cycles[index][1]
        node_num = link_info_all_cycles[index][2]
        node_delay = link_info_all_cycles[index][3]
        delay_sum = link_info_all_cycles[index][4]
        net.configLinkStatus('s0', 's1', 'up')
        last_routing_path = current_routing_path
        last_node_delay = current_node_delay
        current_routing_path = new_routing_path
        current_node_delay = node_delay
        user1_net_dish_sat_delay = None
        user2_net_dish_sat_delay = None
        if last_node_delay == ["NULL"]:
            user1_net_dish_sat_delay = (net, "s0", "s1", 3 * current_node_delay[0])
            user2_net_dish_sat_delay = (net, 's%s' % (intermediate_hop_num - 1), 's%s' % (intermediate_hop_num - 2), 3 * current_node_delay[-1])
        else:
            if current_routing_path[1] != last_routing_path[1]:
                handover_delay1 = 3 * last_node_delay[0] + 3 * current_node_delay[0]
                user1_net_dish_sat_delay = (net, "s0", "s1", handover_delay1)
            if current_routing_path[-2] != last_routing_path[-2]:
                handover_delay2 = 3 * last_node_delay[-1] + 3 * current_node_delay[-1]
                user2_net_dish_sat_delay = (net, 's%s' % (intermediate_hop_num - 1), 's%s' % (intermediate_hop_num - 2), handover_delay2)
        end_handover_nodes = []
        if user1_net_dish_sat_delay is not None:
            end_handover_nodes.append(user1_net_dish_sat_delay)
        if user2_net_dish_sat_delay is not None:
            end_handover_nodes.append(user2_net_dish_sat_delay)
        handover_arr = compute_link_delta(last_routing_path, current_routing_path)
        handover_nodes_group1 = []
        handover_nodes_group2 = []
        for i in range(0, len(handover_arr)):
            if handover_arr[i] == 1:
                if not handover_nodes_group1 or handover_nodes_group1[-1][1] != i - 1:
                    handover_nodes_group1.append((net, i))
                else:
                    handover_nodes_group2.append((net, i))
        print("cycle %s: intermediate handover nodes:" % index, handover_nodes_group1 + handover_nodes_group2)
        print("cycle %s: end handover nodes:" % index, end_handover_nodes)
        if handover_nodes_group1 or handover_nodes_group2 or end_handover_nodes:
            if prediction_delays[index] >= 0:
                print("handover report for cycle %s with %s delay" % (str(index), str(prediction_delays[index])))
                report_handover_status_asynchronously(1, prediction_delays[index])
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor: 
            executor.map(simulate_link_break, handover_nodes_group1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor: 
            executor.map(simulate_link_break, handover_nodes_group2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor: 
            executor.map(simulate_dish_gateway_sat_handover, end_handover_nodes)
        dish_gateway_link_weight1 = node_delay[0]
        set_link_properties(net, "s0", "s1", 150, '%ss' % dish_gateway_link_weight1, max_queue_size=switch_queue_size)
        dish_gateway_link_weight2 = node_delay[-1]
        set_link_properties(net, 's%s' % (intermediate_hop_num - 2), 's%s' % (intermediate_hop_num - 1), 150, '%ss' % dish_gateway_link_weight2, max_queue_size=switch_queue_size)
        for current_hop_idx in range(0, len(current_routing_path) - 3):
            weight = node_delay[current_hop_idx + 1]
            set_link_properties(net, "s%s" % (current_hop_idx + 1), "s%s" % str(current_hop_idx + 2), bent_pipe_link_bandwidth, '%ss' % weight, max_queue_size=switch_queue_size)
        for extra_hop_idx in range(len(current_routing_path) - 3, (intermediate_hop_num - 3)):
            set_link_properties(net, "s%s" % (extra_hop_idx + 1), "s%s" % str(extra_hop_idx + 2), bent_pipe_link_bandwidth, unitialized_bent_pipe_delay, max_queue_size=switch_queue_size)
        tend = time.time()
        sleep_duration = frame_length - (tend - tstart)
        if sleep_duration < 0:
            sleep_duration = 0
        time.sleep(sleep_duration)

topos = {'mytopo': (lambda: MyTopo())}

def read_link_info(input_file_name):
    global intermediate_hop_num
    link_info_all_cycles = []
    in_file = open(f"{LEOEM_INSTALL_FOLDER}/precomputed_paths/{input_file_name}", "r")
    for line in in_file:
        values = line.split("$")
        if len(values) < 5:
            cycle_read = literal_eval(values[0])
            link_info_all_cycles.append([cycle_read, "NULL"])
        else:
            cycle_read = literal_eval(values[0])
            routing_path_read = literal_eval(values[1])
            node_num_read = literal_eval(values[2])
            if node_num_read > intermediate_hop_num:
                intermediate_hop_num = node_num_read
            node_delay_read = literal_eval(values[3])
            delay_sum_read = literal_eval(values[4])
            link_info_all_cycles.append([cycle_read, routing_path_read, node_num_read, node_delay_read, delay_sum_read])
    in_file.close()
    return link_info_all_cycles

def extract_iperf_json_to_csv(out_path):
    import glob, os, json, csv, re
    # Create the CSVs folder if it doesn't exist.
    csvs_folder = os.path.join(out_path, "csvs")
    os.makedirs(csvs_folder, exist_ok=True)
    
    # -------------------------
    # Process client JSON files
    # -------------------------
    client_files = glob.glob(os.path.join(out_path, "iperf_c*.json"))
    client_data = {}
    for fp in client_files:
        try:
            with open(fp, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error parsing {fp}: {e}")
            continue
        fname = os.path.basename(fp)
        m = re.search(r'_(c|x)(\d+)', fname)
        if m:
            flow = int(m.group(2))
            try:
                off = start_times[flow - 1]
            except Exception:
                off = 0
            cid = str(flow)
        else:
            off = 0
            cid = "unknown"
        intervals = data.get("intervals", [])
        for interval in intervals:
            sdata = interval.get("sum", {})
            # Choose the sender stream if available; otherwise, take the first stream.
            sender = None
            for stream in interval.get("streams", []):
                if stream.get("sender", False):
                    sender = stream
                    break
            if sender is None and interval.get("streams"):
                sender = interval["streams"][0]
            # Use the 'end' time from the sum as the time stamp (plus offset)
            t_val = sdata.get("end", 0)
            new_t = float(t_val) + off
            transferred = sdata.get("bytes", 0) / (1024 * 1024)
            bandwidth = sdata.get("bits_per_second", 0) / 1e6
            # Get retransmits (check sender first, else fall back to sum)
            retr = sender.get("retransmits", sdata.get("retransmits", 0))
            cwnd = sender.get("snd_cwnd", 0)
            # Use 'srtt' if present; otherwise, use 'rtt'. Convert from microseconds to milliseconds.
            srtt = sender.get("srtt", sender.get("rtt", 0)) / 1000.0
            rttvar = sender.get("rttvar", 0) / 1000.0

            row = {"time": new_t,
                   "transferred": transferred,
                   "bandwidth": bandwidth,
                   "retr": retr,
                   "cwnd": cwnd,
                   "srtt": srtt,
                   "rttvar": rttvar}
            client_data.setdefault(cid, []).append(row)
    
    # Write out the client CSV files (named as c<flow>.csv)
    for cid, rows in client_data.items():
        csv_fname = os.path.join(csvs_folder, f"c{cid}.csv")
        with open(csv_fname, "w", newline="") as csvfile:
            fieldnames = ["time", "transferred", "bandwidth", "retr", "cwnd", "srtt", "rttvar"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Extracted CSV for client {cid} saved to {csv_fname}")
    
    # -------------------------
    # Process server JSON files
    # -------------------------
    server_files = glob.glob(os.path.join(out_path, "iperf_x*.json"))
    server_data = {}
    for fp in server_files:
        try:
            with open(fp, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error parsing {fp}: {e}")
            continue
        fname = os.path.basename(fp)
        m = re.search(r'_(c|x)(\d+)', fname)
        if m:
            flow = int(m.group(2))
            try:
                off = start_times[flow - 1]
            except Exception:
                off = 0
            sid = str(flow)
        else:
            off = 0
            sid = "unknown"
        intervals = data.get("intervals", [])
        for interval in intervals:
            sdata = interval.get("sum", {})
            t_val = sdata.get("end", 0)
            new_t = float(t_val) + off
            transferred = sdata.get("bytes", 0) / (1024 * 1024)
            bandwidth = sdata.get("bits_per_second", 0) / 1e6
            row = {"time": new_t,
                   "transferred": transferred,
                   "bandwidth": bandwidth}
            server_data.setdefault(sid, []).append(row)
    
    # Write out the server CSV files (named as x<flow>.csv)
    for sid, rows in server_data.items():
        csv_fname = os.path.join(csvs_folder, f"x{sid}.csv")
        with open(csv_fname, "w", newline="") as csvfile:
            fieldnames = ["time", "transferred", "bandwidth"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Extracted CSV for server {sid} saved to {csv_fname}")

# --- New function to extract Astraea output files ---
def extract_astraea_output_to_csv(in_filepath, out_csv_filepath, offset=0):

    # Read all lines from file.
    with open(in_filepath, "r") as infile:
        raw_lines = [line.strip() for line in infile if line.strip()]
    
    data_lines = []
    # If a "----START----" marker is present, only take lines after it until an "----END----" is encountered.
    if any("----START----" in line for line in raw_lines):
        in_section = False
        for line in raw_lines:
            if "----START----" in line:
                in_section = True
                continue
            if "----END----" in line:
                in_section = False
                continue
            if in_section:
                data_lines.append(line)
    else:
        data_lines = raw_lines

    if not data_lines:
        print("No data found in", in_filepath)
        return

    # Determine file type by looking at the first data line.
    first_fields = data_lines[0].split(",")
    if len(first_fields) == 2:
        header = ["time", "bandwidth"]
    elif len(first_fields) == 14:
        header = ["time", "min_rtt", "avg_urtt", "cnt", "srtt", "bandwidth",
                  "thr_cnt", "pacing_rate", "loss_bytes", "packets_out", "retr",
                  "max_packets_out", "cwnd", "CWND"]
    else:
        print("Unexpected number of fields in file:", in_filepath)
        return

    # Compute the minimum timestamp from the first column of all data lines.
    min_time = None
    parsed_lines = []
    for line in data_lines:
        parts = line.split(",")
        try:
            ts = float(parts[0])
        except Exception as e:
            print("Error parsing timestamp from line:", line, e)
            continue
        parsed_lines.append(parts)
        if min_time is None or ts < min_time:
            min_time = ts

    if min_time is None:
        print("Could not determine minimum timestamp in", in_filepath)
        return

    # Adjust the first field (timestamp) for each row.
    adjusted_rows = []
    for parts in parsed_lines:
        try:
            ts = float(parts[0])
            rel_time = (ts - min_time) / 1000.0  + offset# convert ms to seconds
        except Exception as e:
            rel_time = parts[0]
        adjusted_rows.append([rel_time] + parts[1:])

    # Write out the CSV file.
    with open(out_csv_filepath, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        for row in adjusted_rows:
            writer.writerow(row)
    print(f"Extracted Astraea CSV saved to {out_csv_filepath}")




def plot_all_mn(path: str) -> None:
    fig, axs = plt.subplots(7, 1, figsize=(16, 36))
    
    csvs_folder = os.path.join(path, "csvs")
    client_csv_files = glob.glob(os.path.join(csvs_folder, "c*.csv"))
    
    # Sort files by flow number (assuming filenames like c1.csv, c2.csv, etc.)
    client_csv_files.sort(key=lambda f: int(re.search(r'c(\d+)', os.path.basename(f)).group(1)))
    
    for client_file in client_csv_files:
        basename = os.path.basename(client_file)
        flow_num = re.search(r'c(\d+)', basename).group(1)
        server_file = os.path.join(csvs_folder, f"x{flow_num}.csv")
        
        # Read client and server CSV files
        df_client = pd.read_csv(client_file)
        try:
            df_ss_client = pd.read_csv(os.path.join(csvs_folder, f"c{flow_num}_ss.csv"))
        except FileNotFoundError:
            df_ss_client = pd.DataFrame()
        try:
            df_server = pd.read_csv(server_file)
        except Exception as e:
            print(f"Could not read server file {server_file}: {e}")
            continue
        
        # Panel 0: Goodput (server bandwidth)
        axs[0].plot(df_server['time'], df_server['bandwidth'], label=f'x{flow_num} Goodput')
        axs[0].set_title("Goodput (Mbps)")
        axs[0].set_ylabel("Goodput (Mbps)")
        
        # Panel 1: RTT (from client, try 'srtt' column)
        if 'srtt' in df_client.columns:
            axs[1].plot(df_client['time'], df_client['srtt'], label=f'c{flow_num} RTT')
            axs[1].set_title("RTT (ms)")
        elif not df_ss_client.empty and 'srtt' in df_ss_client.columns:
            axs[1].plot(df_ss_client['time'], df_ss_client['srtt'], label=f'c{flow_num} RTT')
            axs[1].set_title("RTT (ms)")
        axs[1].set_ylabel("RTT (ms)")
        
        # Panel 2: Throughput (client bandwidth)
        axs[2].plot(df_client['time'], df_client['bandwidth'], label=f'c{flow_num} Throughput')
        axs[2].set_title("Throughput (Mbps)")
        axs[2].set_ylabel("Throughput (Mbps)")
        
        # Panel 3: CWND (prefer SS file if exists)
        if not df_ss_client.empty and 'cwnd' in df_ss_client.columns:
            axs[3].plot(df_ss_client['time'], df_ss_client['cwnd'], label=f'c{flow_num} CWND')
            axs[3].set_title("CWND (packets)")
        elif 'cwnd' in df_client.columns:
            axs[3].plot(df_client['time'], df_client['cwnd'], label=f'c{flow_num} CWND')
            axs[3].set_title("CWND (packets)")
        axs[3].set_ylabel("CWND (packets)")
        
        # Panel 4: Retransmits
        if 'retr' in df_client.columns:
            axs[4].plot(df_client['time'], df_client['retr'], label=f'c{flow_num} Retransmits')
            axs[4].set_title("Retransmits (packets)")
        axs[4].set_ylabel("Retransmits")
    
    # Process queue files if the "queues" folder exists
    queue_dir = os.path.join(path, "queues")
    if os.path.exists(queue_dir):
        # Optionally, extract a queue limit from the folder name (or set a default)
        match = re.search(r"_(\d+)pkts_", path)
        queue_limit = int(match.group(1)) if match else 100
        axs[5].axhline(queue_limit, color='red', linestyle='--', label='Queue Limit')
        for queue_file in os.listdir(queue_dir):
            if queue_file.endswith('.txt'):
                queue_path = os.path.join(queue_dir, queue_file)
                df_queue = pd.read_csv(queue_path)
                df_queue['time'] = pd.to_numeric(df_queue['time'], errors='coerce')
                df_queue['time'] = df_queue['time'] - df_queue['time'].min()
                df_queue['root_pkts'] = (df_queue['root_pkts']
                                         .str.replace('b', '')
                                         .str.replace('K', '000')
                                         .str.replace('M', '000000')
                                         .str.replace('G', '000000000')
                                         .astype(float))
                df_queue['root_pkts'] = df_queue['root_pkts'] / 1500
                df_queue['interval_drops'] = df_queue['root_drp'].diff().fillna(0)
                axs[5].plot(df_queue['time'], df_queue['root_pkts'], label=f'{queue_file} - Queue Size')
                axs[5].set_title("Queue Size (packets)")
                axs[6].plot(df_queue['time'], df_queue['interval_drops'], linestyle='--', label=f'{queue_file} - Drops')
                axs[6].set_title("Queue Drops (packets)")
    
    # For each subplot, set labels, grid, and adjust limits
    for ax in axs:
        ax.set_xlabel("Time (s)")
        ax.legend(loc='upper left')
        ax.grid(True)
        # Dynamically adjust x limits based on data in the axis
        all_x = []
        for line in ax.get_lines():
            all_x.extend(line.get_xdata())
        if all_x:
            ax.set_xlim(0, max(all_x))
    
    plt.tight_layout(rect=[0, 0, 1, 1], pad=1.0)
    output_file = os.path.join(path, "emulation_results.pdf")
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()

def main():
    current_directory = os.getcwd()
    TOTALDURATION = duration_total
    orca_flow_counter = 0
    astraea_flows_counter = 0

    def start_orca_client(server, outpath, server_ip, start_time):
        threading.Timer(start_time, lambda: server.cmd(
            f'sudo -u {USERNAME} {current_directory}/CC/Orca/receiver.sh {server_ip} 4444 0 {current_directory}/CC/Orca > {outpath}orca_{server.name}.txt &')).start()

    def start_orca_server(client, outpath, start_time, duration):
        threading.Timer(start_time, lambda: client.cmd(
            f'sudo -u {USERNAME} EXPERIMENT_PATH={out_path} {current_directory}/CC/Orca/sender.sh 4444 {orca_flow_counter} {duration} {current_directory}/CC/Orca > {outpath}orca_{client.name}.txt &')).start()
        nonlocal orca_flow_counter
        orca_flow_counter += 1

    def start_astraea_client(server, outpath, start_time, server_ip, duration):
        nonlocal astraea_flows_counter
        cmd = f'sudo -u {USERNAME} {ASTRAEA_INSTALL_FOLDER}/src/build/bin/client_eval --ip={server_ip} --port=5555 --cong=astraea --interval=20 --terminal-out --pyhelper={ASTRAEA_INSTALL_FOLDER}/python/infer.py --model={ASTRAEA_INSTALL_FOLDER}/models/py/ --duration={duration} --id={astraea_flows_counter} > {outpath}/{server.name}.csv &'
        print(cmd)
        threading.Timer(start_time, lambda: server.cmd(cmd)).start() 
        astraea_flows_counter += 1

    def start_astraea_server(client, outpath, start_time):
        cmd = f'sudo -u {USERNAME} {ASTRAEA_INSTALL_FOLDER}/src/build/bin/server --port=5555 --perf-interval=1000 --one-off --terminal-out > {outpath}/{client.name}.csv &'
        print(cmd)
        threading.Timer(start_time, lambda: client.cmd(cmd)).start()

    def start_server(server, outpath, start_time):
        threading.Timer(start_time, lambda: server.cmd(
            f'iperf3 -s --one-off --json -i 1 > {outpath}/iperf_{server.name}.json &')).start()

    def start_client(client, outpath, server_ip, start_time, duration):
        threading.Timer(start_time, lambda: client.cmd(
            f'iperf3 -c {server_ip} -t {duration} -C {protocol} -i 0.1 --json > {outpath}/iperf_{client.name}.json &')).start()

    def start_ping(client, outpath, server_ip, start_time, duration=10):
        threading.Timer(start_time, lambda: client.cmd(
            f'ping {server_ip} -i 0.1 -w {duration} > {outpath}ping_{client.name}.txt &')).start()

    my_logger.info("START MININET LEO SATELLITE NETWORK SIMULATION!")
    links = read_link_info(path_info_file)
    
    out_path = f"{HOME_DIR}/cctestbed/LeoEM/resutls_single_flow/{path_info_file.split('.')[0]}_{bent_pipe_link_bandwidth}mbit_{switch_queue_size}pkts_{len(start_times)}flows_{protocol}/run{run}" 
    rmdirp(out_path)
    mkdirp(out_path)
    my_topo = MyTopo()
    print("create the network")
    net = Mininet(topo=my_topo, link=TCLink, controller=None, xterms=False, autoSetMacs=True)
    net.addController("c0",
                      controller=RemoteController,
                      ip=REMOTE_CONTROLLER_IP,
                      port=6633)
    print("start the network...")
    net.start()
    initialize_link(net)
    
    clients = [net.get(f'c{i+1}') for i in range(NODES)]
    servers = [net.get(f'x{i+1}') for i in range(NODES)]

    for i, server in enumerate(servers):
        print(f"Scheduling iperf server on {server} to start at {start_times[i]} seconds")
        if protocol == 'orca':
            start_orca_client(server, out_path, clients[i].IP(), start_times[i])
        elif protocol == 'astraea':
            start_astraea_client(server, out_path, start_times[i], clients[i].IP(), TOTALDURATION - start_times[i])
        else:
            start_server(server, out_path, start_times[i])

    for i, client in enumerate(clients):
        print(f"Scheduling iperf client on {client} to start at {start_times[i]} seconds")
        if protocol == 'orca':
            start_orca_server(client, out_path, start_times[i], TOTALDURATION - start_times[i])
        elif protocol == 'astraea':
            start_astraea_server(client, out_path, start_times[i])
        else:
            start_client(client, out_path, servers[i].IP(), start_times[i], TOTALDURATION - start_times[i])

    print("start dynamic link simulation")
    update_precomputed_link(links, net)
    
    net.stop()
    
    if protocol == 'astraea':
        csvs_folder = os.path.join(out_path, "csvs")
        os.makedirs(csvs_folder, exist_ok=True)
        astraea_files_client = sorted(glob.glob(os.path.join(out_path, "c*.csv")))
        for i, file in enumerate(astraea_files_client):
            base = os.path.basename(file)
            new_base = "x" + base[1:]        
            out_csv_file = os.path.join(csvs_folder, new_base)
            extract_astraea_output_to_csv(file, out_csv_file, start_times[i])
        astraea_files_server = sorted(glob.glob(os.path.join(out_path, "x*.csv")))
        for i, file in enumerate(astraea_files_server):
            base = os.path.basename(file)
            new_base = "c" + base[1:]
            out_csv_file = os.path.join(csvs_folder, new_base)
            extract_astraea_output_to_csv(file, out_csv_file, start_times[i])

    

    else:
        extract_iperf_json_to_csv(out_path)

    plot_all_mn(out_path)
if __name__ == '__main__':
    main()
