const { Schema, model } = require('mongoose');

const CarroSchema = new Schema({
  title: String,
  price: String,
  caracteristicas: [{ title: String, desc: String }],
  opcionais: [String],
  link: String
});

module.exports = model('Carro', CarroSchema);