const pg = require("pg");
const fs = require("fs");

(async () => {
  const config = {
    host: "localhost",
    user: "postgres",
    password: "123456",
    database: "TCC",
    port: 5432,
  };
  // const client = new pg.Client(config);

  // client.connect((err) => {
  //   if (err) throw err;
  // });

  loadCambio();

})();

function loadCambio() {
  const data = fs.readFileSync("csvFiles/MapCambio.csv", "utf8");
  data.split("\n").forEach()
}

function loadCambustivel() {
  fs.readFile("csvFiles/Map.csv", (err, data) => {

  })
}

function loadCor() {
  fs.readFile("csvFiles/Map.csv", (err, data) => {

  })
}
function loadDirecao() {
  fs.readFile("csvFiles/Map.csv", (err, data) => {

  })
}

function loadMarca() {
  fs.readFile("csvFiles/Map.csv", (err, data) => {

  })
}

function loadModelo() {
  fs.readFile("csvFiles/Map.csv", (err, data) => {

  })
}