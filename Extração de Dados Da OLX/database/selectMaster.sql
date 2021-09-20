select 
	modelo.id as modelo, marca.id as marca, tipoveiculo.id as tipoveiculo, carros.ano, carros.quilometragem, carros.potenciamotor, 
  combustivel.id as combustivel, cambio.id as cambio, direcao.id as direcao, cor.id as cor, carros.portas, carros.finalplaca, 
  carros.vidroeletrico, carros.travaeletrica, carros.arcondicionado, carros.direcaohidraulica, carros.som, carros.airbag, carros.alarme, 
  carros.sensorre, carros.camerare, carros.blindado, carros.valor 
from carros 
join modelo on modelo.description = carros.modelo
join marca on marca.description = carros.marca
join tipoveiculo on tipoveiculo.description = carros.tipoveiculo
join combustivel on combustivel.description = carros.combustivel
join cambio on cambio.description = carros.cambio
join direcao on direcao.description = carros.direcao
join cor on cor.description = carros.cor where carros.ano > 2016;