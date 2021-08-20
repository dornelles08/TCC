const pg = require("pg");
const fs = require("fs");
const config = require("./config/db");
const listFilesDirectory = require("./utils/ListFilesDirectory");
const ProgressBar = require('progress');

/**
 * Cria tabelas relacionadas aos atributos que são string 
 * Insere no banco de dados (postgres) a relação de codigo (int) com o valor (string)
*/
const TransformarDados = async () => {
  const client = new pg.Client(config);

  client.connect((err) => {
    if (err) throw err;
  });

  const files = await listFilesDirectory("Mapeamento");
  const bar = new ProgressBar('Transforming [:bar] :percent :etas', {
    complete: '=',
    incomplete: ' ',
    width: 20,
    total: files.length - 1
  });
  files.forEach(file => {
    loadInfo(file, client);
    bar.tick();
  })

}

function loadInfo(file, db) {

  let data = fs.readFileSync(file, "utf8");
  data = data.split("\n");
  data.pop();
  data = data.map(d => (d.split(",")));

  const bar = new ProgressBar('Saving DB [:bar] :percent :etas', {
    complete: '=',
    incomplete: ' ',
    width: 20,
    total: data.length - 1
  });

  const tableName = file.split('/')[1].replace("Map", "").split(".")[0].toLowerCase();
  db.query(`CREATE TABLE IF NOT EXISTS ${tableName}( 
    id integer primary key, 
    description varchar(100) 
  )`);

  data.forEach((options) => {
    const id = parseInt(options[0]);
    const desc = options[1];
    db.query(`select id from ${tableName} where id = '${id}'`)
      .then(result => {
        bar.tick();
        if (result.rows.length === 0) {
          db.query(`INSERT INTO ${tableName} VALUES ('${id}', '${desc}')`);
        }
      })
  });
}

TransformarDados();