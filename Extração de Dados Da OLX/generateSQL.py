ano_restriction = int(input("Ano minimo: "))

modelo = input("Alguma restrição de modelo? ")

marca = input("Alguma restrição de marca? ")

info_join = [
  "modelo",
  "marca",
  "tipoveiculo",
  "combustivel",
  "cambio",
  "direcao",
  "cor",
]

info = [
  "ano",
  "quilometragem",
  "potenciamotor",
  "portas",
  "finalplaca",
  "vidroeletrico",
  "travaeletrica",
  "arcondicionado",
  "direcaohidraulica",
  "som",
  "airbag",
  "alarme",
  "sensorre",
  "camerare",
  "blindado",
  "valor",
]

join_base = "join * on *.description = carros.* "

select_base = " from carros "

for i in info_join:
  new_join = join_base.replace('*', i)
  select_base += '\n' + new_join

attrs = ""

for i in info_join:
  attrs += i+'.id as '+i+', '

for i in info:
  if i == 'valor':
    attrs += 'carros.'+i
  else:
    attrs += 'carros.'+i+', '

select_base.replace('-', attrs)

if ano_restriction > 0:
  select_base += '\n' + f'where carros.ano > {ano_restriction}'

if marca != "":
  select_base += '\n' + f'and carros.marca like "%{marca}%"'
if modelo != "":
  select_base += '\n' + f'and carros.modelo like \'%{modelo}%\''

select = 'select ' + attrs + select_base

print(select)
