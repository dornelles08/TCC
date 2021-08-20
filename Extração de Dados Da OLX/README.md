# Introdução

Scripts para carregar e transformar os dados coletados da olx e armazenados no MongoDB.

## Instalação

Para executar os scripts precisa do node instalado.

Para instalar as dependências execute:

```bash
npm i
```

## Uso

Carregar e transformar os dados corretamente é necessário executar os arquivos na sequência correta. A sequência é:

1. MongoToCsv.js
2. CsvToPostgres.js
3. ConvertStringToCode.js
4. TransdormandoDados.js

```bash
node MongoToCsv.js
node CsvToPostgres.js
node ConvertStringToCode.js
node TransdormandoDados.js
```
