const puppeteer = require('puppeteer');
const mongoose = require('mongoose');

const Anuncios = require('../model/Anuncios');
const Carro = require('../model/Carro');

(async () => {
  await mongoose.connect('mongodb://localhost:27017/TCCsp', {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  });

  const browser = await puppeteer.launch({ timeout: 500000 });
  const page = await browser.newPage();

  for (let i = 1; i <= 100; i++) {
    try {
      await page.goto(
        `https://sp.olx.com.br/sao-paulo-e-regiao/autos-e-pecas/carros-vans-e-utilitarios?o=${i}`
      );
      //`https://se.olx.com.br/autos-e-pecas/carros-vans-e-utilitarios?o=${i}`

      const linkAnuncios = await page.evaluate(() => {
        const nodeList = document.querySelector('#ad-list');
        const itens = nodeList.querySelectorAll('a');
        const carrosArray = [...itens];

        const linksAnuncios = carrosArray.map((anuncio) => {
          return {
            link: anuncio.getAttribute('href'),
          };
        });
        return linksAnuncios;
      });

      linkAnuncios.forEach(async (link) => {
        await Anuncios.create(link);
        // const existe = await Carro.findOne({ link: link.link });
        // if (!existe) {
        // }
      });
      console.log(`Pagina ${i}`);
    } catch (error) {
      console.log(error.message);
    }
  }

  await browser.close();

  console.log('Finalizado');
})();
