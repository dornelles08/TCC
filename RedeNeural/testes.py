import pandas as pd

carros = pd.read_csv('testeSk.csv')

# print(carros[carros['marca'] == 'PEUGEOT'].shape)

peugeot = carros[carros['marca'] == 'PEUGEOT']

peugeot.to_csv('peugeot.csv', index=False)
