const mongoose = require("mongoose");

const Anuncios = require("../model/Anuncios");
const coletarDados = require("./coletaDados");

(async () => {
  await mongoose.connect("mongodb+srv://geral:geral@cluster0.sg3qs.mongodb.net/TCC?retryWrites=true&w=majority", {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  });

  let interval = setInterval(async () => {
    try {
      const anuncios = await Anuncios.find().limit(1);

      if (anuncios.length == 0) {
        console.log("Finalizado");
        clearInterval(interval);
      }

      anuncios.forEach(async (a) => {
        const dados = await coletarDados(a.link);
        console.log(dados);
        await Anuncios.deleteOne({ link: a.link });
      });
    } catch (error) {
      console.log(error.message);
    }
  }, 20000);
})();