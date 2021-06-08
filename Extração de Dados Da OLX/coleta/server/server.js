const express = require('express')
const mongoose = require("mongoose");
const Carro = require('./model/Carro')
const Anuncios = require('./model/Anuncios')

const app = express()

app.use(express.json())

app.post('/saveOnMongo', async (req, res) => {
  console.log(new Date());
  await mongoose.connect("mongodb+srv://geral:geral@cluster0.sg3qs.mongodb.net/TCC?retryWrites=true&w=majority", {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  });
  try {
    const dadosAnuncio = req.body;
    const existe = await Carro.findOne({ link: dadosAnuncio.link });
    if (!existe) {
      await Carro.create(dadosAnuncio);
      res.send()
    } else {
      res.status(400).send('carro já existe')
    }
  } catch (error) {
    console.log(error.message);
    res.status(500).send(error.message)
  }
})

app.get('/onMongo', async (req, res) => {
  console.log(new Date());
  await mongoose.connect("mongodb+srv://geral:geral@cluster0.sg3qs.mongodb.net/TCC?retryWrites=true&w=majority", {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  });
  try {
    const dadosAnuncio = req.body;
    const carros = await Carro.find({});
    const anuncios = await Anuncios.find({});
    return res.json({ carros: carros.length, anuncios: anuncios.length })
  } catch (error) {
    console.log(error.message);
    res.status(500).send(error.message)
  }
})

app.listen(1997, () => { console.log('Running'); })