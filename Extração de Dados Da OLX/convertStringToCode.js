const pg = require('pg');
const fs = require("fs");

(async () => {
  const config = {
    host: "localhost",
    user: "postgres",
    password: "123456",
    database: "TCC",
    port: 5432,
  };

  const client = new pg.Client(config);

  client.connect((err) => {
    if (err) throw err;
  });

  fs.writeFile("MapCor.csv", `index,cor\n`, { flag: "w" }, (err) => { if (err) console.log(err.message); })

  client.query('select distinct cor from carrosv2').then((res => {
    res.rows.forEach((linha, index) => {
      console.log(`${index},${linha.cor}`);      
      fs.writeFile("MapCor.csv", `${index},${linha.cor}\n`, { flag: "a" }, (err) => { if (err) console.log(err.message); })
    });
  }));
})();