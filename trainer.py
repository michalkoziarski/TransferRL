import tensorflow as tf
import json
import cPickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from gridworld import *
from model import Network
from collections import deque
from shutil import copyfile


class Trainer:
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name', time.strftime('%Y_%m_%d_%H-%M-%S', time.gmtime()))
        self.curriculum_name = kwargs.get('curriculum_name', None)
        self.display_flag = kwargs.get('display_flag', False)
        self.verbose = kwargs.get('verbose', True)

        self.root_path = 'models'
        self.default_params_path = 'params.json'
        self.default_world_path = kwargs.get('world_path', 'world.json')
        self.results_path = os.path.join(self.root_path, self.model_name)
        self.model_path = os.path.join(self.results_path, 'model.ckpt')
        self.checkpoint_path = os.path.join(self.results_path, 'checkpoint')
        self.params_path = os.path.join(self.results_path, 'params.json')
        self.world_path = os.path.join(self.results_path, self.default_world_path)
        self.replay_memory_path = os.path.join(self.results_path, 'replay_memory.pickle')
        self.plot_path = os.path.join(self.results_path, 'rewards.png')
        self.episode_log_path = os.path.join(self.results_path, 'episodes.log')
        self.frame_log_path = os.path.join(self.results_path, 'frames.log')

        self.sess = tf.InteractiveSession()

        if os.path.exists(self.results_path):
            self.load()
        else:
            self.initialize()

        if self.display_flag:
            self.display = Display(width=self.params['width'], height=self.params['height'])

    def initialize(self):
        if self.verbose:
            print 'Initializing model %s...' % self.model_name

        if not os.path.exists(self.root_path):
            os.mkdir(self.root_path)

        os.mkdir(self.results_path)

        with open(self.default_params_path) as f:
            self.params = json.load(f)

        with open(self.default_world_path) as f:
            self.world = json.load(f)

        self.entities = {}

        for k, v in self.world.iteritems():
            self.entities[eval(k)] = v

        self.params['model_name'] = self.model_name
        self.params['current_episode'] = 0
        self.params['current_frame'] = 0
        self.params['current_exploration_rate'] = self.params['initial_exploration_rate']

        self.replay_memory = deque()
        self.reward_history = []

        with open(self.frame_log_path, 'w') as f:
            f.write('frame,reward\n')

        with open(self.episode_log_path, 'w') as f:
            f.write('episode,reward\n')

        with open(self.params_path, 'w') as f:
            json.dump(self.params, f, indent=2, separators=(',', ': '))

        with open(self.world_path, 'w') as f:
            json.dump(self.world, f, indent=2, separators=(',', ': '))

        with open(self.replay_memory_path, 'wb') as f:
            cPickle.dump(self.replay_memory, f)

        self.init_tf()

        if self.curriculum_name:
            curriculum_path = os.path.join(self.root_path, self.curriculum_name)

            copyfile(os.path.join(curriculum_path, 'model.ckpt'), self.model_path)
            copyfile(os.path.join(curriculum_path, 'checkpoint'), self.checkpoint_path)

            self.sess.run(tf.initialize_all_variables())
            self.restore()
        else:
            self.sess.run(tf.initialize_all_variables())
            self.save()

    def load(self):
        if self.verbose:
            print 'Loading model %s...' % self.model_name

        with open(self.params_path) as f:
            self.params = json.load(f)

        with open(self.world_path) as f:
            self.world = json.load(f)

        self.entities = {}

        for k, v in self.world.iteritems():
            self.entities[eval(k)] = v

        with open(self.replay_memory_path, 'rb') as f:
            self.replay_memory = cPickle.load(f)

        self.reward_history = list(pd.read_csv(self.episode_log_path)['reward'])

        self.init_tf()
        self.restore()

    def init_tf(self):
        self.network = Network(input_shape=[self.params['width'], self.params['height'], self.params['memory']],
                               output_shape=[self.params['actions']])

        self._actions = tf.placeholder(tf.float32, [None, self.params['actions']])
        self._rewards = tf.placeholder(tf.float32, [None])
        self._predicted_rewards = tf.reduce_sum(tf.mul(self.network.output, self._actions), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self._rewards - self._predicted_rewards))
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate']).minimize(self.cost)
        self.saver = tf.train.Saver()

    def save(self):
        self.saver.save(self.sess, self.model_path)

    def restore(self):
        self.saver.restore(self.sess, self.model_path)

    def train(self):
        while self.params['current_frame'] < self.params['frames']:
            gw = GridWorld(entities=self.entities, width=self.params['width'], height=self.params['height'])

            while True:
                state = gw.state(memory=self.params['memory'])
                predicted_rewards = self.network.output.eval(feed_dict={self.network.state: state})

                if self.display_flag and self.params['current_episode'] % self.params['display_step'] == 0:
                    self.display.draw(gw, predicted_rewards[0])
                    time.sleep(0.01)

                if random.random() <= self.params['current_exploration_rate']:
                    action = random.randrange(self.params['actions'])
                else:
                    action = np.argmax(predicted_rewards)

                reward = gw.act(action)
                next_state = gw.state(memory=self.params['memory'])

                with open(self.frame_log_path, 'a') as f:
                    f.write('%d,%.2f\n' % (self.params['current_frame'], reward))

                terminal = (True if (gw.terminal() or gw.t() >= self.params['episode_length']) else False)

                self.replay_memory.append((state, action, reward, next_state, terminal))

                if len(self.replay_memory) >= self.params['replay_memory_size']:
                    self.replay_memory.popleft()

                if self.params['current_frame'] >= self.params['replay_start']:
                    batch = random.sample(self.replay_memory, self.params['batch_size'])

                    states = [b[0] for b in batch]
                    rewards = [b[2] for b in batch]
                    actions = []

                    for i in range(len(batch)):
                        actions.append(np.zeros([self.params['actions']]))
                        actions[i][batch[i][1]] = 1

                        if not batch[i][4]:
                            rewards[i] += self.params['reward_decay'] * np.max(
                                self.network.output.eval(feed_dict={self.network.state: batch[i][3]}))

                    states = np.reshape(states, [-1, self.params['width'], self.params['height'],
                                                 self.params['memory']])

                    self.train_step.run(feed_dict={self._actions: actions, self._rewards: rewards,
                                                   self.network.state: states})

                self.params['current_frame'] += 1

                if self.params['current_frame'] <= self.params['exploration_rate_decay']:
                    self.params['current_exploration_rate'] = self.params['initial_exploration_rate'] + \
                                                              self.params['current_frame'] * \
                                                              (self.params['final_exploration_rate'] -
                                                               self.params['initial_exploration_rate']) \
                                                              / float(self.params['exploration_rate_decay'])

                if terminal:
                    with open(self.episode_log_path, 'a') as f:
                        f.write('%d,%.2f\n' % (self.params['current_episode'], gw.total_reward()))

                    self.reward_history.append(gw.total_reward())

                    if self.params['current_episode'] % self.params['display_step'] == 0:
                        self.plot()

                    break

            print 'Episode #%d: total reward of %.2f in %d steps, with exploration rate %.2f' % \
                  (self.params['current_episode'], gw.total_reward(), gw.t(), self.params['current_exploration_rate'])

            if self.params['current_episode'] % self.params['save_step'] == 0:
                print 'Saving model...'

                with open(self.params_path, 'w') as f:
                    json.dump(self.params, f, indent=2, separators=(',', ': '))

                with open(self.replay_memory_path, 'wb') as f:
                    cPickle.dump(self.replay_memory, f)

                self.save()

            self.params['current_episode'] += 1

        self.save()
        self.plot()

        with open(self.params_path, 'w') as f:
            json.dump(self.params, f, indent=2, separators=(',', ': '))

        with open(self.replay_memory_path, 'wb') as f:
            cPickle.dump(self.replay_memory, f)

    def plot(self, window=5000):
        episodes = range(window / 2, len(self.reward_history) - window / 2)
        means = [np.mean(self.reward_history[(i - window / 2):(i + window / 2)]) for i in episodes]

        if len(means) > 0:
            plt.figure()
            plt.plot(episodes, means)
            plt.xlabel('episode')
            plt.ylabel('reward')
            plt.savefig(self.plot_path)
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name')
    parser.add_argument('--curriculum_name')
    parser.add_argument('--world_path')
    parser.add_argument('--display')
    parser.add_argument('--verbose')

    trainer = Trainer(**vars(parser.parse_args()))
    trainer.train()
