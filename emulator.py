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

if "SUDO_USER" in os.environ:
    USERNAME = os.environ["SUDO_USER"]
    HOME_DIR = os.path.expanduser(f"~{USERNAME}")
else:
    HOME_DIR = os.path.expanduser("~")
    USERNAME = os.path.basename(HOME_DIR)

my_logger = logging.getLogger('MyLogger')
my_logger.setLevel(logging.DEBUG)

handler = logging.handlers.SysLogHandler(address = '/dev/log')

my_logger.addHandler(handler)

REMOTE_CONTROLLER_IP = "127.0.0.1"
kernel_output = open('/dev/kmsg', 'w')
status_report_delayed_shift = -0.5


if len(sys.argv) != 7:
    print(f"6 args:  path_info_file, start_times, bw, queue_size, protocol, run {len(sys.argv)}")
    exit()

intermediate_hop_num = 0
frame_length = 1

path_info_file = sys.argv[1]
start_times = list(map(float, sys.argv[2].strip('[]').split(',')))
bent_pipe_link_bandwidth = [3]
unitialized_bent_pipe_delay = '0.01ms'
switch_queue_size = int(sys.argv[4])
protocol = sys.argv[5]
run = sys.argv[6]


NODES = 1
USER = 'mihai'

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
    t = threading.Thread(target = report_handover_status, args = (handover_status, delay), daemon = True)
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
            clients.append(self.addHost(f'c{i+1}', ip=f"10.0.{i+1}.101/24")) # , inNamespace=False
            servers.append(self.addHost(f'x{i+1}', ip=f"10.0.{i+1}.102/24"))

        for i in range(intermediate_hop_num):
            switches.append(self.addSwitch('s%s' % i))

        for i in range(intermediate_hop_num - 1):
            self.addLink('s%s' % i, 's%s' % (i + 1), bw=bent_pipe_link_bandwidth, delay=unitialized_bent_pipe_delay, max_queue_size=switch_queue_size)

        for i in range(NODES):
            self.addLink(f"c{i+1}", "s0", bw=bent_pipe_link_bandwidth, delay=unitialized_bent_pipe_delay, max_queue_size=switch_queue_size)
            self.addLink(f"x{i+1}", "s%s" % (intermediate_hop_num - 1), bw=bent_pipe_link_bandwidth, delay=unitialized_bent_pipe_delay,  max_queue_size=switch_queue_size)

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
        set_link_properties(net, f"c{i+1}", "s0", bent_pipe_link_bandwidth, unitialized_bent_pipe_delay, max_queue_size=switch_queue_size)
        set_link_properties(net, f"x{i+1}", f"s{intermediate_hop_num - 1}", bent_pipe_link_bandwidth, unitialized_bent_pipe_delay, max_queue_size=switch_queue_size)

def update_precomputed_link(link_info_all_cycles, net):
    print("cycle numbers:", len(link_info_all_cycles))
    global current_routing_path, last_routing_path, last_node_delay, current_node_delay

    data_path = 'satcp/report_timing_error/comparsion_1_2/latency_deltas.mat'
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

topos = { 'mytopo': ( lambda: MyTopo() ) }

def read_link_info(input_file_name):
    global intermediate_hop_num

    link_info_all_cycles = []
    in_file = open("precomputed_paths/" + input_file_name, "r")
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

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if not (os.path.isdir(path) and os.path.exists(path)):
            raise
def rmdir(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        if not (os.path.isdir(path) and os.path.exists(path)):
            raise


def main():
    current_directory = os.getcwd()
    TOTALDURATION = 20
    orca_flow_counter = 0
    astraea_flows_counter = 0

    def start_orca_client(server, outpath, server_ip, start_time):
        threading.Timer(start_time, lambda: server.cmd(f'sudo -u {USER} {current_directory}/CC/Orca/receiver.sh {server_ip} 4444 0 {current_directory}/CC/Orca > {outpath}orca_{server.name}.txt &')).start()

    def start_orca_server(client, outpath, start_time, duration):
        threading.Timer(start_time, lambda: client.cmd(f'sudo -u {USER} EXPERIMENT_PATH={outpath} {current_directory}/CC/Orca/sender.sh 4444 {orca_flow_counter} {duration} {current_directory}/CC/Orca > {outpath}orca_{client.name}.txt &')).start()
        nonlocal orca_flow_counter # ??????????????????????????????????
        orca_flow_counter += 1

    def start_astraea_client(server, outpath, start_time, server_ip, duration):
        nonlocal astraea_flows_counter
        cmd = f'sudo -u {USER} {current_directory}/CC/astraea-open-source/src/build/bin/client_eval --ip={server_ip} --port=5555 --cong=astraea --interval=20 --terminal-out --pyhelper={current_directory}/CC/astraea-open-source/python/infer.py --model={current_directory}/CC/astraea-open-source/models/py/ --duration={duration} --id={astraea_flows_counter} > {outpath}astraea_{server.name}.txt &'
        print(cmd)
        threading.Timer(start_time, lambda: server.cmd(cmd)).start() 
        astraea_flows_counter += 1
    def start_astraea_server(client, outpath, start_time):
        cmd = f'sudo -u {USER} {current_directory}/CC/astraea-open-source/src/build/bin/server --port=5555  --perf-interval=1000 --one-off --terminal-out > {outpath}astraea_{client.name}.txt &'
        print(cmd)
        threading.Timer(start_time, lambda: client.cmd(cmd)).start()


    def start_server(server, outpath, start_time):
        threading.Timer(start_time, lambda: server.cmd(f'iperf3 -s --one-off --json  -i 1 > {outpath}iperf_{server.name}.json &')).start()

    def start_client(client, outpath, server_ip, start_time, duration):
        threading.Timer(start_time, lambda: client.cmd(f'iperf3 -c {server_ip} -t {duration} -C {protocol} -i 0.1 --json > {outpath}iperf_{client.name}.json &')).start()

    def start_ping(client, outpath, server_ip, start_time, duration=10):
        threading.Timer(start_time, lambda: client.cmd(f'ping {server_ip} -i 0.1 -w {duration} > {outpath}ping_{client.name}.txt &')).start()

    # python3 ~/pox/pox.py misc.learning_switch
    # sudo python3.8 emulator.py Starlink_NY_LDN_15_ISL_path.log
    #
    #
    #       Starlink_NY_LDN_15_ISL_path.log
    #       Starlink_SD_NY_15_BP_path.log
    #       Starlink_SD_NY_15_ISL_path.log
    #       Starlink_SD_SEA_15_BP_path.log
    #       Starlink_SD_Shanghai_15_ISL_path.log
    #       Starlink_SEA_NY_15_BP_path.log
    #


    my_logger.info("START MININET LEO SATELLITE NETWORK SIMULATION!")
    links = read_link_info(path_info_file)
    out_path = f"{path_info_file.split('.')[0]}/QSize_{switch_queue_size}/2_flows/algo_{protocol}/run_{run}/"
    path = f"{HOME_DIR}/cctestbed/LeoEM/resutls_single_flow/{path_info_file.split('.')[0]}_{bent_pipe_link_bandwidth}mbit_{switch_queue_size}pkts_{len(start_times)}flows_{protocol}/run{run}" 
    mkdir(out_path)
    my_topo = MyTopo()
    print("create the network")
    net = Mininet(topo=my_topo, link=TCLink, controller=None, xterms=False, host=CPULimitedHost, autoPinCpus=True, autoSetMacs=True)
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
            start_orca_server(client, out_path,  start_times[i], TOTALDURATION - start_times[i])
        elif protocol == 'astraea':
            start_astraea_server(client, out_path, start_times[i])
        else:
            start_client(client, out_path, servers[i].IP(), start_times[i], TOTALDURATION - start_times[i])



    # threading.Timer(TOTALDURATION + 3, lambda: net.stop()).start()
    print("start dynamic link simulation")
    update_precomputed_link(links, net)
    
    CLI(net)
    net.stop()
    

if __name__ == '__main__':
    main()
