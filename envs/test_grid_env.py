from envs.env import TrafficSimulator, PhaseMap, PhaseSet
from config.test.build_file import gen_rou_file
import numpy as np


PHASE_NUM = 8
STATE_NAMES = ['wave', 'wait']


class TestGridPhase(PhaseMap):
    def __init__(self):
        # 共8个相位
        phases = ['GGrGrrGGrGrr', 'GrrGGrGrrGGr', 'GrGGrrGrGGrr', 'GrrGrGGrrGrG',
                  'GGGGrrGrrGrr', 'GrrGGGGrrGrr', 'GrrGrrGGGGrr', 'GrrGrrGrrGGG']
        self.phases = {PHASE_NUM: PhaseSet(phases)}


class TestGridController:
    def __init__(self, node_names):
        self.name = 'greedy'
        self.node_names = node_names

    def forward(self, obs):
        actions = []
        for ob, node_name in zip(obs, self.node_names):
            actions.append(self.greedy(ob, node_name))

    def greedy(self, ob, node_name):
        # hard code the mapping from state to number of cars
        flows = [ob[0] + ob[3], ob[2] + ob[5], ob[1] + ob[4],
                 ob[1] + ob[2], ob[4] + ob[5]]
        return np.argmax(np.array(flows))


class TestGridEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        self.peak_flow1 = config.getint('peak_flow1')
        self.peak_flow2 = config.getint('peak_flow2')
        self.init_density = config.getfloat('init_density')
        super().__init__(config, output_path, is_record, record_stat, port=port)

    def _init_node_phase_id(self, node_name):
        return PHASE_NUM

    def _init_neighbor_map(self):
        neighbor_map = {}
        for node_cnt in [3, 4, 7, 8]:
            node_east = 'node_' + str(node_cnt+1)
            node_west = 'node_' + str(node_cnt-1)
            node_north = 'node_' + str(node_cnt-3)
            node_south = 'node_' + str(node_cnt+3)
            neighbor_map['node_' + str(node_cnt)] = [node_north, node_east, node_south, node_west]
        return neighbor_map

    def _init_distance_map(self):
        distance_map = {}
        distance_map['node_3'] = {}
        distance_map['node_4'] = {}
        distance_map['node_7'] = {}
        distance_map['node_8'] = {}
        return distance_map

    def _init_map(self):
        self.neighbor_map = self._init_neighbor_map()
        self.distance_map = self._init_distance_map()
        self.max_distance = 0
        self.phase_map = TestGridPhase()
        self.state_names = STATE_NAMES

    def _init_sim_config(self, seed):
        return gen_rou_file(self.data_path,
                            self.peak_flow1,
                            self.peak_flow2,
                            self.init_density,
                            seed=seed,
                            thread=self.sim_thread)

    def plot_stat(self, rewards):
        self.state_stat['reward'] = rewards


