const pg = require('pg');
const fs = require("fs");
const config = require("./config/db");
const ProgressBar = require('progress');

/** 
 * Tranforma todos os atributos do tipo string 
 * e converte para numerico
 */
const ConvertStringToCode = async () => {
  console.log("3 - Tranforma todos os atributos do tipo string e converte para numerico");
  const client = new pg.Client(config);

  client.connect((err) => {
    if (err) throw err;
  });

  const dir = "./Mapeamento";
  if (!fs.existsSync(dir))
    fs.mkdirSync(dir);

  //Marca
  client.query('select distinct marca from carros').then((res => {
    const bar = new ProgressBar('Converting [:bar] :percent :etas', {
      complete: '=',
      incomplete: ' ',
      width: 20,
      total: res.rows.length - 1
    });
    fs.writeFileSync("Mapeamento/MapMarca.csv", "", (err) => console.log(err.message))
    res.rows.forEach((linha, index) => {
      fs.writeFile("Mapeamento/MapMarca.csv", `${index},${linha.marca}\n`, { flag: "a" }, (err) => { if (err) console.log(err.message); bar.tick() })
    });
  }));

  //Tipo do Veiculo
  client.query('select distinct tipoveiculo from carros').then((res => {
    const bar = new ProgressBar('Converting [:bar] :percent :etas', {
      complete: '=',
      incomplete: ' ',
      width: 20,
      total: res.rows.length - 1
    });
    fs.writeFileSync("Mapeamento/MapTipoVeiculo.csv", "", (err) => console.log(err.message))
    res.rows.forEach((linha, index) => {
      fs.writeFile("Mapeamento/MapTipoVeiculo.csv", `${index},${linha.tipoveiculo}\n`, { flag: "a" }, (err) => { if (err) console.log(err.message); bar.tick() })
    });
  }));

  //Combustivel
  client.query('select distinct combustivel from carros').then((res => {
    const bar = new ProgressBar('Converting [:bar] :percent :etas', {
      complete: '=',
      incomplete: ' ',
      width: 20,
      total: res.rows.length - 1
    });
    fs.writeFileSync("Mapeamento/MapCombustivel.csv", "", (err) => console.log(err.message))
    res.rows.forEach((linha, index) => {
      fs.writeFile("Mapeamento/MapCombustivel.csv", `${index},${linha.combustivel}\n`, { flag: "a" }, (err) => { if (err) console.log(err.message); bar.tick() })
    });
  }));

  //Cambio
  client.query('select distinct cambio from carros').then((res => {
    const bar = new ProgressBar('Converting [:bar] :percent :etas', {
      complete: '=',
      incomplete: ' ',
      width: 20,
      total: res.rows.length - 1
    });
    fs.writeFileSync("Mapeamento/MapCambio.csv", "", (err) => console.log(err.message))
    res.rows.forEach((linha, index) => {
      fs.writeFile("Mapeamento/MapCambio.csv", `${index},${linha.cambio}\n`, { flag: "a" }, (err) => { if (err) console.log(err.message); bar.tick() })
    });
  }));

  //Direção
  client.query('select distinct direcao from carros').then((res => {
    const bar = new ProgressBar('Converting [:bar] :percent :etas', {
      complete: '=',
      incomplete: ' ',
      width: 20,
      total: res.rows.length - 1
    });
    fs.writeFileSync("Mapeamento/MapDirecao.csv", "", (err) => console.log(err.message))
    res.rows.forEach((linha, index) => {
      fs.writeFile("Mapeamento/MapDirecao.csv", `${index},${linha.direcao}\n`, { flag: "a" }, (err) => { if (err) console.log(err.message); bar.tick() })
    });
  }));

  //Cor
  client.query('select distinct cor from carros').then((res => {
    const bar = new ProgressBar('Converting [:bar] :percent :etas', {
      complete: '=',
      incomplete: ' ',
      width: 20,
      total: res.rows.length - 1
    });
    fs.writeFileSync("Mapeamento/MapCor.csv", "", (err) => console.log(err.message))
    res.rows.forEach((linha, index) => {
      fs.writeFile("Mapeamento/MapCor.csv", `${index},${linha.cor}\n`, { flag: "a" }, (err) => { if (err) console.log(err.message); bar.tick() })
    });
  }));

  //Modelo
  client.query('select distinct modelo from carros').then((res => {
    const bar = new ProgressBar('Converting [:bar] :percent :etas', {
      complete: '=',
      incomplete: ' ',
      width: 20,
      total: res.rows.length - 1
    });
    fs.writeFileSync("Mapeamento/MapModelo.csv", "", (err) => console.log(err.message))
    res.rows.forEach((linha, index) => {
      fs.writeFile("Mapeamento/MapModelo.csv", `${index},${linha.modelo}\n`, { flag: "a" }, (err) => { if (err) console.log(err.message); bar.tick() })
      if (index === res.rows.length - 1) {
        console.log("Fim");
        client.end();
      }
    });
  }));

}

ConvertStringToCode();