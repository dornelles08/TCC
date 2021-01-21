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
    const opcionais = carro.caracteristicas;
    opcionais.forEach(({ title }) => {
      if (opcionaisGlobal.indexOf(title) == -1) {
        opcionaisGlobal.push(title)
      }
    });
  });

  console.log(opcionaisGlobal);
})();