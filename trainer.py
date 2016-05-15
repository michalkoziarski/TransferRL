import tensorflow as tf
import json
import cPickle
import os
import matplotlib.pyplot as plt
import pandas as pd

from gridworld import *
from model import Network
from collections import deque


sess = tf.InteractiveSession()

if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name = time.strftime('%Y_%m_%d_%H-%M-%S', time.gmtime())

if len(sys.argv) > 2:
    display_flag = bool(sys.argv[2])
else:
    display_flag = False

if os.path.exists(os.path.join('models', model_name)):
    print 'Loading model %s...' % model_name

    with open(os.path.join('models', model_name, 'params.json')) as f:
        params = json.load(f)

    with open(os.path.join('models', model_name, 'replay_memory.pickle'), 'rb') as f:
        replay_memory = cPickle.load(f)

    network = Network(input_shape=[params['width'], params['height'], params['memory']],
                      output_shape=[params['actions']])

    _actions = tf.placeholder(tf.float32, [None, params['actions']])
    _rewards = tf.placeholder(tf.float32, [None])
    _predicted_rewards = tf.reduce_sum(tf.mul(network.output, _actions), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(_rewards - _predicted_rewards))
    train_step = tf.train.RMSPropOptimizer(learning_rate=params['learning_rate'], decay=params['learning_decay'],
                                           momentum=params['learning_momentum']).minimize(cost)

    saver = tf.train.Saver()
    saver.restore(sess, os.path.join('models', model_name, 'model.ckpt'))

    reward_history = list(pd.read_csv(os.path.join('models', model_name, 'episodes.log'))['reward'])
else:
    print 'Initializing model %s...' % model_name

    if not os.path.exists('models'):
        os.mkdir('models')

    os.mkdir(os.path.join('models', model_name))

    with open('params.json') as f:
        params = json.load(f)

    params['model_name'] = model_name
    params['episode'] = 0
    params['frame'] = 0
    params['exploration_rate'] = params['initial_exploration_rate']

    replay_memory = deque()

    network = Network(input_shape=[params['width'], params['height'], params['memory']],
                      output_shape=[params['actions']])

    _actions = tf.placeholder(tf.float32, [None, params['actions']])
    _rewards = tf.placeholder(tf.float32, [None])
    _predicted_rewards = tf.reduce_sum(tf.mul(network.output, _actions), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(_rewards - _predicted_rewards))
    train_step = tf.train.RMSPropOptimizer(learning_rate=params['learning_rate'], decay=params['learning_decay'],
                                           momentum=params['learning_momentum']).minimize(cost)

    sess.run(tf.initialize_all_variables())

    with open(os.path.join('models', model_name, 'frames.log'), 'w') as f:
        f.write('frame,reward\n')

    with open(os.path.join('models', model_name, 'episodes.log'), 'w') as f:
        f.write('episode,reward\n')

    with open(os.path.join('models', model_name, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2, separators=(',', ': '))

    with open(os.path.join('models', model_name, 'replay_memory.pickle'), 'wb') as f:
        cPickle.dump(replay_memory, f)

    saver = tf.train.Saver()
    saver.save(sess, os.path.join('models', model_name, 'model.ckpt'))

    reward_history = []

if display_flag or params['display']:
    display = Display(width=params['width'], height=params['height'])

while params['frame'] < params['frames']:
    gw = GridWorld(entities={Goal: 1}, width=params['width'], height=params['height'])

    while True:
        state = gw.state(memory=params['memory'])
        predicted_rewards = network.output.eval(feed_dict={network.state: state})

        if (display_flag or params['display']) and params['episode'] % params['display_step'] == 0:
            display.draw(gw, predicted_rewards[0])
            time.sleep(0.01)

        if random.random() <= params['exploration_rate']:
            action = random.randrange(params['actions'])
        else:
            action = np.argmax(predicted_rewards)

        reward = gw.act(action)
        next_state = gw.state(memory=params['memory'])

        with open(os.path.join('models', model_name, 'frames.log'), 'a') as f:
            f.write('%d,%.2f\n' % (params['frame'], reward))

        terminal = (True if (gw.terminal() or gw.t() >= params['episode_length']) else False)

        replay_memory.append((state, action, reward, next_state, terminal))

        if len(replay_memory) >= params['replay_memory_size']:
            replay_memory.popleft()

        if params['frame'] >= params['replay_start']:
            batch = random.sample(replay_memory, params['batch_size'])

            states = [b[0] for b in batch]
            rewards = [b[2] for b in batch]
            actions = []

            for i in range(len(batch)):
                actions.append(np.zeros([params['actions']]))
                actions[i][batch[i][1]] = 1

                if not batch[i][4]:
                    rewards[i] += params['reward_decay'] * np.max(network.output.eval(feed_dict={network.state: batch[i][3]}))

            states = np.reshape(states, [-1, params['width'], params['height'], params['memory']])

            train_step.run(feed_dict={_actions: actions, _rewards: rewards, network.state: states})

        params['frame'] += 1

        if terminal:
            with open(os.path.join('models', model_name, 'episodes.log'), 'a') as f:
                f.write('%d,%.2f\n' % (params['episode'], gw.total_reward()))

            reward_history.append(gw.total_reward())

            if params['episode'] % params['display_step'] == 0:
                plt.figure()
                plt.plot(range(1, len(reward_history) + 1), reward_history)
                plt.xlabel('episode')
                plt.ylabel('reward')
                plt.savefig(os.path.join('models', model_name, 'rewards.png'))
                plt.close()

            break

    print 'Episode #%d: total reward of %.2f in %d steps, with exploration rate %.2f' % \
          (params['episode'], gw.total_reward(), gw.t(), params['exploration_rate'])

    if params['episode'] % params['save_step'] == 0:
        print 'Saving model...'

        with open(os.path.join('models', model_name, 'params.json'), 'w') as f:
            json.dump(params, f, indent=2, separators=(',', ': '))

        with open(os.path.join('models', model_name, 'replay_memory.pickle'), 'wb') as f:
            cPickle.dump(replay_memory, f)

        saver.save(sess, os.path.join('models', model_name, 'model.ckpt'))

    if params['frame'] <= params['exploration_rate_decay']:
        params['exploration_rate'] -= (params['initial_exploration_rate'] - params['final_exploration_rate']) / \
                                      float(params['exploration_rate_decay'])

    params['episode'] += 1
