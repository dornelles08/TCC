const pg = require('pg');

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

  client.query('select distinct tipoveiculo from teste1').then((res => {    
    res.rows.forEach((linha, index) => {
      console.log(linha.tipoveiculo, index);
      client.query(`update teste1 set tipoveiculo = ${index} where tipoveiculo = '${linha.tipoveiculo}'`)
    });
  }))

})();