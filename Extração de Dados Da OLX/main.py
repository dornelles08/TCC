import subprocess
from time import time

inicio = time()
scripts = ["MongoToCsv.js",
           "CsvToPostgres.js",
           "ConvertStringToCode.js",
           "TransdormandoDados.js"]

subprocess.run(["npm", "install"])

for script in scripts:
    subprocess.run(["clear"])
    subprocess.run(["node", script])

fim = time()

print(int(fim-inicio), "s")

exit()
