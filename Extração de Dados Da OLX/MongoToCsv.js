const mongoose = require("mongoose");
const Carro = require("./Carro");
const dbCarro = require("./model/Carro");
const fs = require("fs");
const ProgressBar = require("progress");

/**
 * Baixa todos os carros do banco de dados (mongo)
 * e salva em csv para tratamento dos dados
 */
const MongoToCsv = async () => {
  console.log(
    "1 - Baixa todos os carros do banco de dados (mongo) e salva em csv para tratamento dos dados"
  );
  fs.writeFile(
    "carros.csv",
    "Modelo,Marca,Tipo de veículo,Ano,Quilometragem,Potência do motor,Combustível,Câmbio,Direção,Cor,Portas,Final de placa,Vidro elétrico,Trava elétrica,Ar condicionado,Direção hidráulica,Som,Air bag,Alarme,Sensor de ré,Câmera de ré,Blindado,Valor\n",
    (err) => {
      if (err) console.log(err.message);
    }
  );

  console.log("Conectando no banco");
  const db = await mongoose.connect(
    "mongodb+srv://geral:geral@cluster0.sg3qs.mongodb.net/TCC_SP?retryWrites=true&w=majority",
    {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    }
  );
  console.log("Conecxão estabelecida");

  console.log("Inicio");
  const limit = 1000;
  let skip = 0;
  while (true) {
    const carros = await dbCarro.find().limit(limit).skip(skip);

    const bar = new ProgressBar("Saving [:bar] :percent :etas", {
      complete: "=",
      incomplete: " ",
      width: 20,
      total: carros.length - 1,
    });

    carros.forEach((carro, idx) => {
      const newCarro = new Carro();

      newCarro.setValor(parseFloat(carro.price.substr(3).replace(".", "")));

      separandoOpcionais(carro.opcionais, newCarro);
      separarCaracteristicas(carro.caracteristicas, newCarro);

      newCarro.saveCSV("carros_sp.csv");
      bar.tick();
    });

    if (carros.length == 0) {
      break;
    }

    skip += limit;
  }

  db.disconnect();

  console.log("Fim");
};

function separandoOpcionais(opcionais, carro) {
  carro.setVidroEletrico(0);
  carro.setTravaEletrica(0);
  carro.setArCondicionado(0);
  carro.setDirecaoHidraulica(0);
  carro.setSom(0);
  carro.setAirBag(0);
  carro.setAlarme(0);
  carro.setSensorRe(0);
  carro.setCameraRe(0);
  carro.setBlindado(0);

  opcionais.forEach((op) => {
    if (op === "Vidro elétrico") {
      carro.setVidroEletrico(1);
    } else if (op === "Trava elétrica") {
      carro.setTravaEletrica(1);
    } else if (op === "Ar condicionado") {
      carro.setArCondicionado(1);
    } else if (op === "Direção hidráulica") {
      carro.setDirecaoHidraulica(1);
    } else if (op === "Som") {
      carro.setSom(1);
    } else if (op === "Air bag") {
      carro.setAirBag(1);
    } else if (op === "Alarme") {
      carro.setAlarme(1);
    } else if (op === "Sensor de ré") {
      carro.setSensorRe(1);
    } else if (op === "Câmera de ré") {
      carro.setCameraRe(1);
    } else if (op === "Blindado") {
      carro.setBlindado(1);
    }
  });
}

function separarCaracteristicas(caracteristicas, carro) {
  carro.setModelo(null);
  carro.setMarca(null);
  carro.setTipoVeiculo(null);
  carro.setAno(null);
  carro.setQuilometragem(null);
  carro.setPotenciaMotor(null);
  carro.setCombustivel(null);
  carro.setCambio(null);
  carro.setDirecao(null);
  carro.setCor(null);
  carro.setPortas(null);
  carro.setFinalPlaca(null);

  caracteristicas.forEach((c) => {
    if (c.title === "Modelo") {
      carro.setModelo(c.desc);
    } else if (c.title === "Marca") {
      carro.setMarca(c.desc);
    } else if (c.title === "Tipo de veículo") {
      carro.setTipoVeiculo(c.desc);
    } else if (c.title === "Ano") {
      carro.setAno(parseInt(c.desc));
    } else if (c.title === "Quilometragem") {
      carro.setQuilometragem(parseInt(c.desc));
    } else if (c.title === "Potência do motor") {
      carro.setPotenciaMotor(parseFloat(c.desc));
    } else if (c.title === "Combustível") {
      carro.setCombustivel(c.desc);
    } else if (c.title === "Câmbio") {
      carro.setCambio(c.desc);
    } else if (c.title === "Direção") {
      carro.setDirecao(c.desc);
    } else if (c.title === "Cor") {
      carro.setCor(c.desc);
    } else if (c.title === "Portas") {
      carro.setPortas(parseInt(c.desc.substr(0, 2)));
    } else if (c.title === "Final de placa") {
      carro.setFinalPlaca(parseInt(c.desc));
    }
  });
}

MongoToCsv();
