const pg = require('pg');
const fs = require("fs");
const config = require("./config/db");

/** 
 * Tranforma todos os atributos do tipo string 
 * e converte para numerico
 */
moudle.export = async () => {
  const client = new pg.Client(config);

  client.connect((err) => {
    if (err) throw err;
  });

  client.query('select distinct modelo from carros').then((res => {
    res.rows.forEach((linha, index) => {
      console.log(`${index},${linha.modelo}`);
      fs.writeFile("Mapeamento/MapModelo.csv", `${index},${linha.modelo}\n`, { flag: "a" }, (err) => { if (err) console.log(err.message); })
    });
  }));

  client.query('select distinct marca from carros').then((res => {
    res.rows.forEach((linha, index) => {
      console.log(`${index},${linha.marca}`);
      fs.writeFile("Mapeamento/MapMarca.csv", `${index},${linha.marca}\n`, { flag: "a" }, (err) => { if (err) console.log(err.message); })
    });
  }));

  client.query('select distinct tipoveiculo from carros').then((res => {
    res.rows.forEach((linha, index) => {
      console.log(`${index},${linha.tipoveiculo}`);
      fs.writeFile("Mapeamento/MapTipoVeiculo.csv", `${index},${linha.tipoveiculo}\n`, { flag: "a" }, (err) => { if (err) console.log(err.message); })
    });
  }));

  client.query('select distinct combustivel from carros').then((res => {
    res.rows.forEach((linha, index) => {
      console.log(`${index},${linha.combustivel}`);
      fs.writeFile("Mapeamento/MapCombustivel.csv", `${index},${linha.combustivel}\n`, { flag: "a" }, (err) => { if (err) console.log(err.message); })
    });
  }));

  client.query('select distinct cambio from carros').then((res => {
    res.rows.forEach((linha, index) => {
      console.log(`${index},${linha.cambio}`);
      fs.writeFile("Mapeamento/MapCambio.csv", `${index},${linha.cambio}\n`, { flag: "a" }, (err) => { if (err) console.log(err.message); })
    });
  }));

  client.query('select distinct direcao from carros').then((res => {
    res.rows.forEach((linha, index) => {
      console.log(`${index},${linha.direcao}`);
      fs.writeFile("Mapeamento/MapDirecao.csv", `${index},${linha.direcao}\n`, { flag: "a" }, (err) => { if (err) console.log(err.message); })
    });
  }));

  client.query('select distinct cor from carros').then((res => {
    res.rows.forEach((linha, index) => {
      console.log(`${index},${linha.cor}`);
      fs.writeFile("Mapeamento/MapCor.csv", `${index},${linha.cor}\n`, { flag: "a" }, (err) => { if (err) console.log(err.message); })
    });
  }));
}