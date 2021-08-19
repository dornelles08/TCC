const pg = require("pg");
const fs = require("fs");

(async () => {
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

  loadInfo("csvFiles/MapCambio.csv", client);

})();

function loadInfo(file, db) {
  let data = fs.readFileSync(file, "utf8");
  data = data.split("\n").map(d => (d.split(",")));

  const tableName = file.split('/')[1].replace("Map", "").split(".")[0].toLowerCase();

  db.query(`CREATE TABLE IF NOT EXISTS ${tableName}( 
    id integer primary key, 
    desc varchar(100) 
  )`)
    .then(result => console.log(result))
    .catch(err => console.log(err));

  data.forEach((options) => {
    const id = parseInt(options[0])
    const desc = options[1]
    console.log(options);
  });
}