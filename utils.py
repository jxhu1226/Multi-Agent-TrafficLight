import os
import logging
import time
import subprocess
import itertools
import random
import numpy as np
import pandas as pd
import tensorflow as tf


def init_dir(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dirs = {}
    pathes = ['log', 'data', 'model']
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


def copy_file(src_dir, tar_dir):
    cmd = 'cp %s %s' % (src_dir, tar_dir)
    subprocess.check_call(cmd, shell=True)


def init_test_mode(test_mode):
    if test_mode == 'no_test':
        return False, False
    elif test_mode == 'in_train_test':
        return True, False
    elif test_mode == 'after_train_test':
        return False, True
    else:
        return True, True


class Counter:
    def __init__(self, total_step, test_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False
        # self.init_test = True

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop


class TransBuffer:
    def reset(self):
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)

    def add_transition(self, ob, a, r, *_args, **_kwargs):
        raise NotImplementedError()

    def sample_transition(self, *_args, **_kwargs):
        raise NotImplementedError()


class ReplayBuffer(TransBuffer):
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.cum_size = 0
        self.buffer = []

    def add_transition(self, ob, a, r, next_ob, done):
        experience = (ob, a, r, next_ob, done)
        if self.cum_size < self.buffer_size:
            self.buffer.append(experience)
        else:
            ind = int(self.cum_size % self.buffer_size)
            self.buffer[ind] = experience
        self.cum_size += 1

    def reset(self):
        self.buffer = []
        self.cum_size = 0

    def sample_transition(self):
        # Randomly sample batch_size examples
        minibatch = random.sample(self.buffer, self.batch_size)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])
        return state_batch, action_batch, next_state_batch, reward_batch, done_batch

    @property
    def size(self):
        return min(self.buffer_size, self.cum_size)


class Scheduler:
    def __init__(self, val_init, val_min=0, total_step=0, decay='linear'):
        self.val = val_init
        self.N = float(total_step)
        self.val_min = val_min
        self.decay = decay
        self.n = 0

    def get(self, n_step):
        self.n += n_step
        if self.decay == 'linear':
            return max(self.val_min, self.val * (1 - self.n / self.N))
        else:
            return self.val


class Trainer():
    def __init__(self, env, model, global_counter, summary_writer, run_test, output_path=None):
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.agent = self.env.agent
        self.model = model
        self.sess = self.model.sess
        self.n_step = self.model.n_step
        self.summary_writer = summary_writer
        self.run_test = run_test
        assert self.env.T % self.n_step == 0
        self.data = []
        self.output_path = output_path
        if run_test:
            self.test_num = self.env.test_num
            logging.info('Testing: total test num: %d' % self.test_num)
        self._init_summary()

    def _init_summary(self):
        self.train_reward = tf.placeholder(tf.float32, [])
        self.train_summary = tf.summary.scalar('train_reward', self.train_reward)
        self.test_reward = tf.placeholder(tf.float32, [])
        self.test_summary = tf.summary.scalar('test_reward', self.test_reward)

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            summ = self.sess.run(self.train_summary, {self.train_reward: reward})
        else:
            summ = self.sess.run(self.test_summary, {self.test_reward: reward})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def explore(self, prev_ob, prev_done):
        ob = prev_ob
        done = prev_done
        rewards = []
        for _ in range(self.n_step):
            if self.agent.endswith('a2c'):
                policy, value = self.model.forward(ob, done)
                # need to update fingerprint before calling step
                if self.agent == 'ma2c':
                    self.env.update_fingerprint(policy)
                if self.agent == 'a2c':
                    action = np.random.choice(np.arange(len(policy)), p=policy)
                else:
                    action = []
                    for pi in policy:
                        action.append(np.random.choice(np.arange(len(pi)), p=pi))
            else:
                action, policy = self.model.forward(ob, mode='explore')
            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            global_step = self.global_counter.next()
            self.cur_step += 1
            if self.agent.endswith('a2c'):
                self.model.add_transition(ob, action, reward, value, done)
            else:
                self.model.add_transition(ob, action, reward, next_ob, done)
            # logging
            if self.global_counter.should_log():
                logging.info('''Training: global step %d, episode step %d,
                                   ob: %s, a: %s, pi: %s, r: %.2f, train r: %.2f, done: %r''' %
                             (global_step, self.cur_step,
                              str(ob), str(action), str(policy), global_reward, np.mean(reward), done))
            # # termination
            # if done:
            #     self.env.terminate()
            #     time.sleep(2)
            #     ob = self.env.reset()
            #     self._add_summary(cum_reward / float(self.cur_step), global_step)
            #     cum_reward = 0
            #     self.cur_step = 0
            # else:
            if done:
                break
            ob = next_ob
        if self.agent.endswith('a2c'):
            if done:
                R = 0 if self.agent == 'a2c' else [0] * self.model.n_agent
            else:
                R = self.model.forward(ob, False, 'v')
        else:
            R = 0
        return ob, done, R, rewards

    def perform(self, test_ind, demo=False, policy_type='default'):
        ob = self.env.reset(gui=demo, test_ind=test_ind)
        # note this done is pre-decision to reset LSTM states!
        done = True
        self.model.reset()
        rewards = []
        while True:
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            elif self.agent.endswith('a2c'):
                # policy-based on-poicy learning
                policy = self.model.forward(ob, done, 'p')
                if self.agent == 'ma2c':
                    self.env.update_fingerprint(policy)
                if self.agent == 'a2c':
                    if policy_type != 'deterministic':
                        action = np.random.choice(np.arange(len(policy)), p=policy)
                    else:
                        action = np.argmax(np.array(policy))
                else:
                    action = []
                    for pi in policy:
                        if policy_type != 'deterministic':
                            action.append(np.random.choice(np.arange(len(pi)), p=pi))
                        else:
                            action.append(np.argmax(np.array(pi)))
            else:
                # value-based off-policy learning
                if policy_type != 'stochastic':
                    action, _ = self.model.forward(ob)
                else:
                    action, _ = self.model.forward(ob, stochastic=True)
            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            if done:
                break
            ob = next_ob
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward

    def run_thread(self, coord):
        '''Multi-threading is disabled'''
        ob = self.env.reset()
        done = False
        cum_reward = 0
        while not coord.should_stop():
            ob, done, R, cum_reward = self.explore(ob, done, cum_reward)
            global_step = self.global_counter.cur_step
            if self.agent.endswith('a2c'):
                self.model.backward(R, self.summary_writer, global_step)
            else:
                self.model.backward(self.summary_writer, global_step)
            self.summary_writer.flush()
            if (self.global_counter.should_stop()) and (not coord.should_stop()):
                self.env.terminate()
                coord.request_stop()
                logging.info('Training: stop condition reached!')
                return

    def run(self):
        while not self.global_counter.should_stop():
            # test
            if self.run_test and self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                self.env.train_mode = False
                for test_ind in range(self.test_num):
                    mean_reward, std_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(mean_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'avg_reward': mean_reward,
                           'std_reward': std_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step, is_train=False)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
            # train
            self.env.train_mode = True
            ob = self.env.reset()
            # note this done is pre-decision to reset LSTM states!
            done = True
            self.model.reset()
            self.cur_step = 0
            rewards = []
            while True:
                ob, done, R, cur_rewards = self.explore(ob, done)
                rewards += cur_rewards
                global_step = self.global_counter.cur_step
                if self.agent.endswith('a2c'):
                    self.model.backward(R, self.summary_writer, global_step)
                else:
                    self.model.backward(self.summary_writer, global_step)
                # termination
                if done:
                    self.env.terminate()
                    break
            rewards = np.array(rewards)
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            log = {'agent': self.agent,
                   'step': global_step,
                   'test_id': -1,
                   'avg_reward': mean_reward,
                   'std_reward': std_reward}
            self.data.append(log)
            self._add_summary(mean_reward, global_step)
            self.summary_writer.flush()
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')
