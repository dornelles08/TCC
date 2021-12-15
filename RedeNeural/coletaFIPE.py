import requests
import psycopg2

conn = psycopg2.connect(
    host="10.0.0.185",
    database="TCC",
    user="postgres",
    password="123456")

cur = conn.cursor()

urlMarcas = "http://veiculos.fipe.org.br/api/veiculos/ConsultarMarcas"
urlModelos = "http://veiculos.fipe.org.br/api/veiculos/ConsultarModelos"
urlAnoModelo = "http://veiculos.fipe.org.br/api/veiculos/ConsultarAnoModelo"
urlValor = "http://veiculos.fipe.org.br/api/veiculos/ConsultarValorComTodosParametros"

headers = {
    "Content-Type": "application/json",
    "Host": "veiculos.fipe.org.br",
    "Referer": "http://veiculos.fipe.org.br"
}

data = {
    "codigoTabelaReferencia": 279,
    "codigoTipoVeiculo": 1
}

response = requests.post(urlMarcas, headers=headers, json=data)

marcas = response.json()

file = open("valorFipe.txt", 'w')
fileErr = open("valorFipeErr.txt", 'w')

file.write("Marca;Modelo;Ano;Valor\n")


def marcaExist(marca):
    cur.execute(f"SELECT id FROM marca WHERE description = '{marca}'")
    if cur.rowcount == 0:
        return False
    else:
        return True


for marca in marcas:
    labelMarca = marca["Label"]
    valueMarca = marca["Value"]

    if marcaExist(labelMarca.upper()):
        print(f"Marca: {labelMarca}")
        data["codigoMarca"] = valueMarca
        response = requests.post(urlModelos, headers=headers, json=data)
        modelos = response.json()["Modelos"]

        for modelo in modelos:
            labelModelo = modelo["Label"]
            valueModelo = modelo["Value"]
            data["codigoModelo"] = valueModelo
            response = requests.post(urlAnoModelo, headers=headers, json=data)
            anosModelo = response.json()

            for anoModelo in anosModelo:
                labelAnoModelo = anoModelo["Label"]
                valueAnoModelo = anoModelo["Value"]
                if "Dies" in labelAnoModelo:
                    data["codigoTipoCombustivel"] = 3
                else:
                    data["codigoTipoCombustivel"] = 1

                data["tipoConsulta"] = "tradicional"
                data["anoModelo"] = int(valueAnoModelo.split("-")[0])
                response = requests.post(urlValor, headers=headers, json=data)

                valorModelo = response.json()

                if "erro" not in valorModelo:
                    valor = valorModelo["Valor"].replace(
                        ".", "").split(",")[0].split(" ")[1]
                    marca = valorModelo["Marca"].upper()
                    modelo = valorModelo["Modelo"].upper()
                    anoModelo = valorModelo["AnoModelo"]

                    file.write(f"{marca};{modelo};{anoModelo};{valor}\n")
                else:
                    data["codigoTipoCombustivel"] = 2
                    response = requests.post(
                        urlValor, headers=headers, json=data)
                    valorModelo = response.json()
                    if "erro" not in valorModelo:
                        valor = valorModelo["Valor"].replace(
                            ".", "").split(",")[0].split(" ")[1]
                        marca = valorModelo["Marca"].upper()
                        modelo = valorModelo["Modelo"].upper()
                        anoModelo = valorModelo["AnoModelo"]
                        file.write(f"{marca};{modelo};{anoModelo};{valor}\n")
                    else:
                        print(data)
                        fileErr.write(f"{labelModelo};{valueModelo}")

    else:
        print(f"Marca não existe: {labelMarca}")
