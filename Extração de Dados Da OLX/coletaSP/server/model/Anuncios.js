const { Schema, model } = require('mongoose');

const AnuncioSchema = new Schema({
  link: {
    type: String
  }
});

module.exports = model('Anuncio', AnuncioSchema);