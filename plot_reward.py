import numpy as np
import seaborn as sns
import pandas as pd
import sys
import os


assert len(sys.argv) > 1

model_name = sys.argv[1]

if len(sys.argv) > 2:
    batch = int(sys.argv[2])
else:
    batch = 50000

df = pd.read_csv(os.path.join('models', model_name, 'log.csv'))

assert len(df['reward']) > batch

rewards = [df['reward'].iloc[0]]
index = 1

while index < len(df['reward']):
    k = 1

    while df['frame'].iloc[index] <= df['frame'].iloc[index - k]:
        rewards.pop()
        k += 1

    rewards.append(df['reward'].iloc[index])
    index += 1

mean_rewards = []
mean_frames = []
index = 0

while index + batch <= len(rewards):
    mean_rewards.append(np.mean(rewards[index:(index + batch)]))
    mean_frames.append(index + batch / 2)
    index += batch

sns.tsplot(data=mean_rewards, time=mean_frames)
sns.plt.show()
