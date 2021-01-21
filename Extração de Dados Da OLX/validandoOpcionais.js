const mongoose = require('mongoose');
const Carro = require('./model/Carro');

(async () => {
  await mongoose.connect('mongodb://localhost:27017/TCC', {
    useNewUrlParser: true,
    useUnifiedTopology: true
  });

  const opcionaisGlobal = [];

  const carros = await Carro.find();

  carros.forEach(carro => {
    const opcionais = carro.opcionais.toString().split(',');
    opcionais.forEach(o => {
      if (opcionaisGlobal.indexOf(o) == -1) {
        opcionaisGlobal.push(o)
      }
    });
  });

  console.log(opcionaisGlobal);
})();