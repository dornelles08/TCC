from pymongo import MongoClient

def separarCaracteristicas(caracteristicas, newCarro):
  for c in listCaracteristicas:
    newCarro[c] = None

  for c in caracteristicas:    
    if (c['title'] == 'Modelo'):
      newCarro['Modelo'] = c["desc"].replace(',', ' ')
    elif (c['title'] == 'Marca'):
      newCarro['Marca'] = c["desc"]
    elif (c['title'] == 'Tipo de veículo'):
      newCarro['Tipo de veículo'] = c["desc"]
    elif (c['title'] == 'Ano'):
      newCarro['Ano'] = c["desc"]
    elif (c['title'] == 'Quilometragem'):
      newCarro['Quilometragem'] = c["desc"]
    elif (c['title'] == 'Potência do motor'):
      newCarro['Potência do motor'] = c["desc"]
    elif (c['title'] == 'Combustível'):
      newCarro['Combustível'] = c["desc"]
    elif (c['title'] == 'Câmbio'):
      newCarro['Câmbio'] = c["desc"]
    elif (c['title'] == 'Direção'):
      newCarro['Direção'] = c["desc"]
    elif (c['title'] == 'Cor'):
      newCarro['Cor'] = c["desc"]
    elif (c['title'] == 'Portas'):
      newCarro['Portas'] = c["desc"]
    elif (c['title'] == 'Final de placa'):
      newCarro['Final de placa'] = c["desc"]

def separandoOpcionais(opcionais, newCarro):
  for o in listOpcionais:
    newCarro[o] = 0
  
  for o in opcionais:
    if (o == 'Vidro elétrico'):
      newCarro[o] = 1
    elif (o == 'Trava elétrica'):
      newCarro[o] = 1
    elif (o == 'Ar condicionado'):
      newCarro[o] = 1
    elif (o == 'Direção hidráulica'):
      newCarro[o] = 1
    elif (o == 'Som'):
      newCarro[o] = 1
    elif (o == 'Air bag'):
      newCarro[o] = 1
    elif (o == 'Alarme'):
      newCarro[o] = 1
    elif (o == 'Sensor de ré'):
      newCarro[o] = 1
    elif (o == 'Câmera de ré'):
      newCarro[o] = 1
    elif (o == 'Blindado'):
      newCarro[o] = 1

print("Início")

listCaracteristicas = [
  'Modelo',
  'Marca',
  'Tipo de veículo',
  'Ano',
  'Quilometragem',
  'Potência do motor',
  'Combustível',
  'Câmbio',
  'Direção',
  'Cor',
  'Portas',
  'Final de placa'
]
listOpcionais = [
  'Vidro elétrico',
  'Trava elétrica',
  'Ar condicionado',
  'Direção hidráulica',
  'Som',
  'Air bag',
  'Alarme',
  'Sensor de ré',
  'Câmera de ré',
  'Blindado'
]

uri = "mongodb+srv://geral:geral@cluster0.sg3qs.mongodb.net/TCC_SP?retryWrites=true&w=majority"
client = MongoClient(uri)
db = client['TCC']
collection = db['carros']

arq = open('carros.csv', 'w')
arq.write('Modelo,Marca,Tipo de veículo,Ano,Quilometragem,Potência do motor,Combustível,Câmbio,Direção,Cor,Portas,Final de placa,Vidro elétrico,Trava elétrica,Ar condicionado,Direção hidráulica,Som,Air bag,Alarme,Sensor de ré,Câmera de ré,Blindado,Valor\n')
arq.close()
arq = open('carros.csv', 'a')

print("Buscando Carros - Início")
carros = list(collection.find())
print(len(carros))
print("Buscando Carros - Fim")

for carro in carros:  
  newCarro = {}
  opcionais = carro['opcionais']
  caracteristicas = carro['caracteristicas']

  newCarro['Valor'] = carro['price'][3:].replace('.', '')
  separarCaracteristicas(caracteristicas, newCarro)
  separandoOpcionais(opcionais, newCarro)

  arq.write(f'{newCarro["Modelo"]},{newCarro["Marca"]},{newCarro["Tipo de veículo"]},{newCarro["Ano"]},{newCarro["Quilometragem"]},{newCarro["Potência do motor"]},{newCarro["Combustível"]},{newCarro["Câmbio"]},{newCarro["Direção"]},{newCarro["Cor"]},{newCarro["Final de placa"]},{newCarro["Vidro elétrico"]},{newCarro["Trava elétrica"]},{newCarro["Ar condicionado"]},{newCarro["Direção hidráulica"]},{newCarro["Som"]},{newCarro["Air bag"]},{newCarro["Alarme"]},{newCarro["Sensor de ré"]},{newCarro["Câmera de ré"]},{newCarro["Blindado"]},{newCarro["Valor"]}\n')
  


print("Fim")

arq.close()