import tensorflow as tf
import numpy as np

from gridworld import *
from model import Network
from collections import deque


ACTIONS = 5
WIDTH = 16
HEIGHT = 16
MEMORY = 1
REWARD_DECAY = 0.99
REPLAY_MEMORY_LENGTH = 1000
BATCH_SIZE = 32
FINAL_EXPLORATION_RATE = 0.05
INITIAL_EXPLORATION_RATE = 1.0
EXPLORATION_RATE_DECAY = 500
EPISODE_LENGTH = 256
DISPLAY_STEP = 100
EPISODES = 10000

network = Network([WIDTH, HEIGHT, MEMORY], [ACTIONS])

_actions = tf.placeholder(tf.float32, [None, ACTIONS])
_rewards = tf.placeholder(tf.float32, [None])
_predicted_rewards = tf.reduce_sum(tf.mul(network.output, _actions), reduction_indices=1)
cost = tf.reduce_mean(tf.square(_rewards - _predicted_rewards))
train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

replay_memory = deque()
exploration_rate = INITIAL_EXPLORATION_RATE
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
display = Display()

for episode in range(EPISODES):
    gw = GridWorld(entities={Goal: 1})

    while True:
        if episode % DISPLAY_STEP == 0:
            display.draw(gw)
            time.sleep(0.01)

        state = np.reshape(gw.state(), [-1, WIDTH, HEIGHT, MEMORY])

        if random.random() <= exploration_rate:
            action = random.randrange(ACTIONS)
        else:
            action = np.argmax(network.output.eval(feed_dict={network.state: state}))

        reward = gw.act(action)
        next_state = np.reshape(gw.state(), [-1, WIDTH, HEIGHT, MEMORY])

        terminal = (True if (gw.terminal() or gw.t() >= EPISODE_LENGTH) else False)

        replay_memory.append((state, action, reward, next_state, terminal))

        if len(replay_memory) >= REPLAY_MEMORY_LENGTH:
            replay_memory.popleft()

        if len(replay_memory) >= BATCH_SIZE:
            batch = random.sample(replay_memory, BATCH_SIZE)

            states = [b[0] for b in batch]
            rewards = [b[2] for b in batch]
            actions = []

            for i in range(len(batch)):
                actions.append(np.zeros([ACTIONS]))
                actions[i][batch[i][1]] = 1

                if not batch[i][4]:
                    rewards[i] += REWARD_DECAY * np.max(network.output.eval(feed_dict={network.state: batch[i][3]}))

            states = np.reshape(states, [-1, WIDTH, HEIGHT, MEMORY])

            train_step.run(feed_dict={_actions: actions, _rewards: rewards, network.state: states})

        if terminal:
            break

    print 'Episode #%d: total reward of %.2f in %d steps, with exploration rate %.2f' % \
          (episode, gw.total_reward(), gw.t(), exploration_rate)

    if episode <= EXPLORATION_RATE_DECAY:
        exploration_rate -= (INITIAL_EXPLORATION_RATE - FINAL_EXPLORATION_RATE) / float(EXPLORATION_RATE_DECAY)
