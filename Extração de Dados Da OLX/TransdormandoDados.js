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
  console.log("4 - Cria tabelas relacionadas aos atributos que são string. \nInsere no banco de dados (postgres) a relação de codigo (int) com o valor (string)");
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
  files.forEach((file, index) => {
    loadInfo(file, client, index, files.length)
    bar.tick();
  })

}

function loadInfo(file, db, i, filesLength) {
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

  data.forEach((options, index) => {
    const id = parseInt(options[0]);
    const desc = options[1];
    db.query(`INSERT INTO ${tableName} VALUES ('${id}', '${desc}')`)
      .then(() => {
        bar.tick();
        if (filesLength - 1 === i && data.length - 1 === index) {
          console.log("Fim");
          db.end();
        }
      })
      .catch(() => {
        bar.tick();
        if (filesLength - 1 === i && data.length - 1 === index) {
          console.log("Fim");
          db.end();
        }
      })
  });
}

TransformarDados();