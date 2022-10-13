import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dados = pd.read_csv('dados/sp_completo.csv')
# dados = pd.read_csv('dados/se_completo.csv')
dados.head()

plot = pd.DataFrame(dados.modelo.value_counts().rename_axis(
    'modelo').reset_index(name='counts'))

plt.rc('legend', fontsize=15)
plt.rcParams['font.size'] = '15'

plt.figure(figsize=(16, 8))
sns.barplot(data=plot.query("counts > 150"),
            x='modelo', y="counts", color="blue")
plt.xticks(rotation=45)
plt.xlabel('Modelos', fontsize=20)
plt.ylabel('Frequencia', fontsize=20)
plt.title('Frequencia por Modelo', fontsize=20)
plt.legend()
