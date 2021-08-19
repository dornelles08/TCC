const fs = require("fs");
const pg = require("pg");

const config = {
  host: "10.0.0.185",
  user: "postgres",
  password: "123456",
  database: "TCC",
  port: 5432,
};

const client = new pg.Client(config);

client.connect((err) => {
  if (err) throw err;
});

fs.readFile("carros.csv", "utf8", (err, data) => {
  if (err) console.log(err.message);
  else {
    const linha = data.split("\n");

    for (let i = 1; i < linha.length; i++) {
      const info = linha[i].split(",");
      let dadoNull = false;
      info.forEach((dado) => {
        if (dado == "null") {
          dadoNull = true;
        }
      });
      if (!dadoNull) {
        const query = `
          INSERT INTO carros VALUES 
          ('${info[0]}','${info[1]}','${info[2]}','${info[3]}',
          '${info[4]}','${info[5]}','${info[6]}',
          '${info[7]}','${info[8]}','${info[9]}','${info[10]}',
          '${info[11]}','${info[12]}','${info[13]}',
          '${info[14]}','${info[15]}','${info[16]}',
          '${info[17]}','${info[18]}','${info[19]}','${info[20]}',
          '${info[21]}','${info[22]}')
        `;

        client
          .query(query)
          .then((res) => {})
          .catch((err) => {
            console.log(err.message);
          });
      }
    }
  }
});
