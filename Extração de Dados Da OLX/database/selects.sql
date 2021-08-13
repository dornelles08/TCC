-- Removendo carros antes 2010
delete from carros where ano < 2011

-- Removendo carros antes 2016
delete from carros where ano < 2016

-- Removendo Carros com menos de 1000 km
delete from carros where quilometragem < 1000 and ano < 2020

-- Contagem de Modelos
select count(*), modelo from carros group by modelo order by count(*) desc

-- Possiveis modelos
select count(*) from carros where modelo like '%COROLLA%' --94
select count(*) from carros where modelo like '%GM - CHEVROLET CLASSIC%' --18
select count(*) from carros where modelo like '%GOL%' --138
select count(*) from carros where modelo like '%GOLF%' --28
select count(*) from carros where modelo like '%FIESTA%' --107
select count(*) from carros where modelo like '%HB20%' --135
select count(*) from carros where modelo like '%SANDERO%' --82
select count(*) from carros where modelo like '%UNO%' --54
select count(*) from carros where modelo like '%SIENA%' --78
select count(*) from carros where modelo like '%PALIO%' --82
select count(*) from carros where modelo like '%STRADA%' --62
select count(*) from carros where modelo like '%TORO%' --50
select count(*) from carros where modelo like '%VOYAGE%' --71
select count(*) from carros where modelo like '%CIVIC%'  --32
-------------------------------------------------------------------------------------------------------
