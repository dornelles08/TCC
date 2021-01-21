const fs = require('fs');

class Carro {
  constructor() {
    this.modelo
    this.marca
    this.tipoVeiculo
    this.ano
    this.quilometragem
    this.potenciaMotor
    this.combustivel
    this.cambio
    this.direcao
    this.cor
    this.portas
    this.finalPlaca
    this.vidroEletrico
    this.travaEletrica
    this.arCondicionado
    this.direcaogidraulica
    this.som
    this.airBag
    this.alarme
    this.sensorRe
    this.cameraRe
    this.blindado
    this.valor
  }

  setModelo(modelo) { this.modelo = modelo }
  setMarca(marca) { this.marca = marca }
  setTipoVeiculo(tipoVeiculo) { this.tipoVeiculo = tipoVeiculo }
  setAno(ano) { this.ano = ano }
  setQuilometragem(quilometragem) { this.quilometragem = quilometragem }
  setPotenciaMotor(potenciaMotor) { this.potenciaMotor = potenciaMotor }
  setCombustivel(combustivel) { this.combustivel = combustivel }
  setCambio(cambio) { this.cambio = cambio }
  setDirecao(direcao) { this.direcao = direcao }
  setCor(cor) { this.cor = cor }
  setPortas(portas) { this.portas = portas }
  setFinalPlaca(finalPlaca) { this.finalPlaca = finalPlaca }

  setVidroEletrico(vidroEletrico) { this.vidroEletrico = vidroEletrico }
  setTravaEletrica(travaEletrica) { this.travaEletrica = travaEletrica }
  setArCondicionado(arCondicionado) { this.arCondicionado = arCondicionado }
  setDirecaoHidraulica(direcaohidraulica) { this.direcaohidraulica = direcaohidraulica }
  setSom(som) { this.som = som }
  setAirBag(airBag) { this.airBag = airBag }
  setAlarme(alarme) { this.alarme = alarme }
  setSensorRe(sensorRe) { this.sensorRe = sensorRe }
  setCameraRe(cameraRe) { this.cameraRe = cameraRe }
  setBlindado(blindado) { this.blindado = blindado }

  setValor(valor) { this.valor = valor }

  saveCSV(path) {
    const dados = `${this.modelo},${this.marca},${this.tipoVeiculo},${this.ano},${this.quilometragem},${this.potenciaMotor},${this.combustivel},${this.cambio},${this.direcao},${this.cor},${this.portas},${this.finalPlaca},${this.vidroEletrico},${this.travaEletrica},${this.arCondicionado},${this.direcaohidraulica},${this.som},${this.airBag},${this.alarme},${this.sensorRe},${this.cameraRe},${this.blindado},${this.valor}\n`
    fs.appendFile(path, dados, (err) => { if (err) console.log(err.message); });
  }

  savePostgres(pg) {
    if (this.modelo == null ||
      this.marca == null ||
      this.tipoVeiculo == null ||
      this.ano == null ||
      this.quilometragem == null ||
      this.potenciaMotor == null ||
      this.combustivel == null ||
      this.cambio == null ||
      this.direcao == null ||
      this.cor == null ||
      this.portas == null ||
      this.finalPlaca == null ||
      this.vidroEletrico == null ||
      this.travaEletrica == null ||
      this.arCondicionado == null ||
      this.direcaogidraulica == null ||
      this.som == null ||
      this.airBag == null ||
      this.alarme == null ||
      this.sensorRe == null ||
      this.cameraRe == null ||
      this.blindado == null ||
      this.valor == null) {
      // console.log('Aguma informação está null');
    } else {

    }
    const query =
      `
          INSERT INTO carros VALUES 
          ('${this.modelo}','${this.marca}','${this.tipoVeiculo}','${this.ano}',
          '${this.quilometragem}','${this.potenciaMotor}','${this.combustivel}',
          '${this.cambio}','${this.direcao}','${this.cor}','${this.portas}',
          '${this.finalPlaca}','${this.vidroEletrico}','${this.travaEletrica}',
          '${this.arCondicionado}','${this.direcaohidraulica}','${this.som}',
          '${this.airBag}','${this.alarme}','${this.sensorRe}','${this.cameraRe}',
          '${this.blindado}','${this.valor}')
        `

    pg.query(query).then(res => {

    }).catch(err => {
      console.log(err.message);
    });

  }

  infoNula() {
    if (this.modelo == null ||
      this.marca == null ||
      this.tipoVeiculo == null ||
      this.ano == null ||
      this.quilometragem == null ||
      this.potenciaMotor == null ||
      this.combustivel == null ||
      this.cambio == null ||
      this.direcao == null ||
      this.cor == null ||
      this.portas == null ||
      this.finalPlaca == null ||
      this.vidroEletrico == null ||
      this.travaEletrica == null ||
      this.arCondicionado == null ||
      this.direcaogidraulica == null ||
      this.som == null ||
      this.airBag == null ||
      this.alarme == null ||
      this.sensorRe == null ||
      this.cameraRe == null ||
      this.blindado == null ||
      this.valor == null) {
      return true
    } else {
      return false
    }
  }

}

module.exports = Carro;