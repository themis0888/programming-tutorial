import numpy as np 
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
from scipy import stats
import pdb


# Bar plot 
classes = np.array(list(range(1,11)))

benchmark = np.random.randn(10)
portfolio = np.random.randn(10)

df = pd.DataFrame({'Classes': classes,
                   '#_Wrong': benchmark,
                   '#_Data': portfolio},
                    columns = ['Classes','#_Wrong','#_Data'])

                    
df1 = pd.melt(df, id_vars=['Classes']).sort_values(['variable','value'])
sns.barplot(x='Classes', y='value', hue='variable', data=df1)
# plt.xticks(rotation=90)
plt.ylabel('Error')
plt.title('Err / Num_data per Class')

plt.savefig('sample.png')
plt.cla()




