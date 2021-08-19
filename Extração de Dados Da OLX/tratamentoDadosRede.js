const pg = require("pg");
const fs = require("fs");
const config = require("./config/db");
const listFilesDirectory = require("./ListFilesDirectory");

(async () => {
  const client = new pg.Client(config);

  client.connect((err) => {
    if (err) throw err;
  });

  const files = await listFilesDirectory("Mapeamento");
  console.log(files);
  files.forEach(file => {
    loadInfo(file, client);
  })

})();

function loadInfo(file, db) {
  let data = fs.readFileSync(file, "utf8");
  data = data.split("\n");
  data.pop();
  data = data.map(d => (d.split(",")));


  const tableName = file.split('/')[1].replace("Map", "").split(".")[0].toLowerCase();
  console.log(tableName);
  db.query(`CREATE TABLE IF NOT EXISTS ${tableName}( 
    id integer primary key, 
    description varchar(100) 
  )`);

  data.forEach((options) => {
    const id = parseInt(options[0]);
    const desc = options[1];
    db.query(`select id from ${tableName} where id = '${id}'`)
      .then(result => {
        console.log(result.rows.length);
        if (result.rows.length === 0) {
          db.query(`INSERT INTO ${tableName} VALUES ('${id}', '${desc}')`)
        } else {
          console.log("já cadastrado");
        }
      })
  });
}