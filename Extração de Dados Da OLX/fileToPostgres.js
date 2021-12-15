const fs = require("fs");
const pg = require("pg");
const config = require("./config/db");

(async () => {
  const client = new pg.Client(config);

  client.connect((err) => {
    if (err) throw err;
  });

  const files = [
    "./carro5.txt",
    "./carro10.txt",
    "./carro15.txt",
    "./carro20.txt",
  ];

  await Promise.all(
    files.map(async (file) => {
      let data = fs.readFileSync(file, { encoding: "utf8", flag: "r" });
      data = data.split("\n");
      await Promise.all(
        data.map(async (d) => {
          let carro = d.split(",");
          if (carro.length > 0) {
            let tableName = "";

            if (
              file.replace("./", "").replace(".txt", "").replace('"', "") ==
              "carro5"
            ) {
              tableName = "carros5";
            } else if (
              file.replace("./", "").replace(".txt", "").replace('"', "") ==
              "carro10"
            ) {
              tableName = "carros10";
            } else if (
              file.replace("./", "").replace(".txt", "").replace('"', "") ==
              "carro15"
            ) {
              tableName = "carros15";
            } else if (
              file.replace("./", "").replace(".txt", "").replace('"', "") ==
              "carro20"
            ) {
              tableName = "carros20";
            }

            const query = `INSERT INTO ${tableName} VALUES ('${carro[0]}','${carro[1]}','${carro[2]}','${carro[3]}','${carro[4]}','${carro[5]}','${carro[6]}','${carro[7]}','${carro[8]}','${carro[9]}','${carro[10]}','${carro[11]}','${carro[12]}','${carro[13]}','${carro[14]}','${carro[15]}','${carro[16]}','${carro[17]}','${carro[18]}','${carro[19]}','${carro[20]}','${carro[21]}','${carro[22]}')`;
            client.query(query).catch((e) => {
              console.log(d);
              console.log(e.message);
            });
          }
        })
      );
    })
  );
})();
