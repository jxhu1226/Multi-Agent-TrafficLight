import numpy as np


MAX_CAR_NUM = 30
NET_NAME = 'test'


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def init_routes(density):
    init_flow = '  <flow id="i_%s" departPos="random_free" from="%s" to="%s" begin="0" end="1" departLane="%d" departSpeed="0" number="%d" type="type1"/>\n'
    output = ''
    in_nodes = [1,2,3,3,6,9,9,8,7,7,4,1]
    out_nodes = [i for i in range(1, 13)]
    sink_edges = []
    for i, j in zip(in_nodes, out_nodes):
        node_1 = 'nt' + str(i)
        node_2 = 'np' + str(j)
        sink_edges.append('%s_%s' % (node_1, node_2))

    def get_od(node_1, node_2, k, lane=1):
        source_edge = '%s_%s' % (node_1, node_2)
        sink_edge = np.random.choice(sink_edges)
        return init_flow % (str(k), source_edge, sink_edge, lane, car_num)

    k = 1
    # streets
    car_num = int(MAX_CAR_NUM * density)
    for i in [1,4,7]:
        for j in range(2):
            node_1 = 'nt' + str(i+j)
            node_2 = 'nt' + str(i+j+1)
            output += get_od(node_1, node_2, k)
            k += 1
            output += get_od(node_2, node_1, k)
            k += 1
            output += get_od(node_1, node_2, k, lane=2)
            k += 1
            output += get_od(node_2, node_1, k, lane=2)
            k += 1
    # avenues
    for i in [1, 2, 3]:
        for j in [0, 3]:
            node_1 = 'nt' + str(i + j)
            node_2 = 'nt' + str(i + j + 3)
            output += get_od(node_1, node_2, k)
            k += 1
            output += get_od(node_2, node_1, k)
            k += 1
    return output


def gen_rou_file(path, peak_flow1, peak_flow2, density, seed=None, thread=None):
    sumocfg_file = path + 'test.sumocfg'
    return sumocfg_file


def output_config(thread=None):
    if thread is None:
        out_file = '%s.rou.xml' % NET_NAME
    else:
        out_file = '%s_%d.rou.xml' % (NET_NAME, int(thread))
    str_config = '<configuration>\n  <input>\n'
    str_config += '    <net-file value="%s.net.xml"/>\n' % NET_NAME
    str_config += '    <route-files value="%s"/>\n' % out_file
    str_config += '    <additional-files value="%s.add.xml"/>\n' % NET_NAME
    str_config += '  </input>\n  <time>\n'
    str_config += '    <begin value="0"/>\n    <end value="3600"/>\n'
    str_config += '  </time>\n</configuration>\n'
    return str_config


def output_ild(ild):
    str_adds = '<additional>\n'
    in_nodes = [1, 2, 3, 3, 6, 9, 9, 8, 7, 7, 4, 1]
    out_nodes = [i for i in range(1, 13)]
    # in_edges = [5, 10, 15, 20, 25, 21, 16, 11, 6, 1,
    #             1, 2, 3, 4, 5, 25, 24, 23, 22, 21]
    # out_edges = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20,
    #              1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
    # external edges
    for (i, j) in zip(in_nodes, out_nodes):
        node1 = 'nt' + str(i)
        node2 = 'np' + str(j)
        str_adds += get_ild_str(node2, node1, ild, lane_i=0)
        str_adds += get_ild_str(node2, node1, ild, lane_i=1)
        str_adds += get_ild_str(node2, node1, ild, lane_i=2)
    # streets
    for i in [1, 4, 7]:
        for j in range(2):
            node1 = 'nt' + str(i+j)
            node2 = 'nt' + str(i+j+1)
            str_adds += get_ild_str(node1, node2, ild)
            str_adds += get_ild_str(node2, node1, ild)
            str_adds += get_ild_str(node1, node2, ild, lane_i=1)
            str_adds += get_ild_str(node2, node1, ild, lane_i=1)
            str_adds += get_ild_str(node1, node2, ild, lane_i=2)
            str_adds += get_ild_str(node2, node1, ild, lane_i=2)
    # avenues
    for i in [1,2,3]:
        for j in [0, 3]:
            node1 = 'nt' + str(i + j)
            node2 = 'nt' + str(i + j + 3)
            str_adds += get_ild_str(node1, node2, ild)
            str_adds += get_ild_str(node2, node1, ild)
            str_adds += get_ild_str(node1, node2, ild, lane_i=1)
            str_adds += get_ild_str(node2, node1, ild, lane_i=1)
            str_adds += get_ild_str(node1, node2, ild, lane_i=2)
            str_adds += get_ild_str(node2, node1, ild, lane_i=2)
    str_adds += '</additional>\n'
    return str_adds


def get_ild_str(from_node, to_node, ild_str, lane_i=0):
    edge = '%s_%s' % (from_node, to_node)
    return ild_str % (edge, lane_i, edge, lane_i)


def output_flows(peak_flow1, peak_flow2, density, seed=None):
    '''
    flow1: x11, x12, x13, x14, x15 -> x1, x2, x3, x4, x5
    flow2: x16, x17, x18, x19, x20 -> x6, x7, x8, x9, x10
    flow3: x1, x2, x3, x4, x5 -> x15, x14, x13, x12, x11
    flow4: x6, x7, x8, x9, x10 -> x20, x19, x18, x17, x16
    '''
    if seed is not None:
        print(seed)
        np.random.seed(seed)
    ext_flow = '  <flow id="f_%s" departPos="random_free" from="%s" to="%s" begin="%d" end="%d" vehsPerHour="%d" type="type1"/>\n'
    str_flows = '<routes>\n'
    str_flows += '  <vType id="type1" length="5" accel="5" decel="10"/>\n'
    # initial traffic dist
    if density > 0:
        str_flows += init_routes(density)
    str_flows += '</routes>\n'
    return str_flows


def main():
    # raw.rou.xml file
    write_file('./%s.rou.xml' % NET_NAME, output_flows(1000, 2000, 0.8))
    # add.xml file
    ild = '  <laneAreaDetector file="ild.out" freq="1" id="%s_%d" lane="%s_%d" pos="-50" endPos="-1"/>\n'
    # ild_in = '  <inductionLoop file="ild_out.out" freq="15" id="ild_in:%s" lane="%s_0" pos="10"/>\n'
    write_file('./%s.add.xml' % NET_NAME, output_ild(ild))
    # config file
    write_file('./%s.sumocfg' % NET_NAME, output_config())


if __name__ == '__main__':
    main()