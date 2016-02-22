import seaborn as sns
import pandas as pd
import sys
import os


assert len(sys.argv) > 1

model_name = sys.argv[1]

df = pd.read_csv(os.path.join('models', model_name, 'log.csv'))

sns.tsplot(data=df['reward'], t=df['episode'])
sns.plt.show()
