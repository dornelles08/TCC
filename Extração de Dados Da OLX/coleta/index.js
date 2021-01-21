const mongoose = require('mongoose');

const Anuncios = require('../model/Anuncios');
const Carro = require('../model/Carro');
const coletarDados = require('./coletaDados');

(async () => {
  await mongoose.connect('mongodb://localhost:27017/TCCsp', {
    useNewUrlParser: true,
    useUnifiedTopology: true
  });

  let interval = setInterval(async () => {
    const anuncios = await Anuncios.find().limit(1);

    if (anuncios.length == 0) {
      console.log('Finalizado');
      clearInterval(interval);
    }

    anuncios.forEach(async a => {
      const dados = await coletarDados(a.link);
      console.log(dados);
      await Anuncios.deleteOne({ link: a.link });
    });

  }, 20000);


})();

(async () => {
  await mongoose.connect('mongodb://localhost:27017/TCC', {
    useNewUrlParser: true,
    useUnifiedTopology: true
  });
  const existe = await Carro.findOne({ title: "UP MOVE 1.0 2015" });
  if (existe) {
    console.log(existe);
  } else {
    console.log('Não existe');
  }
});