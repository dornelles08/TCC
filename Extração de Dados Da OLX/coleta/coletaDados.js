const puppeteer = require('puppeteer');
const Carro = require('../model/Carro');

module.exports = async (link) => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  await page.goto(link);

  try {
    const dadosAnuncio = await page.evaluate(() => {
      const title = document.querySelector('h1.sc-1q2spfr-0.fSThqK.sc-ifAKCX.cmFKIN').innerText;

      const price = document.querySelector('h2.sc-ifAKCX.eQLrcK').innerText;

      const caracteristicas = [...document.querySelectorAll('div.sc-bwzfXH.h3us20-0.cBfPri')[2].children];
      const caracteristicasList = caracteristicas.map(c => {
        return {
          title: c.children[0].children[1].children[0].innerText,
          desc: c.children[0].children[1].children[1].innerText
        }
      });

      const opcionais = [...document.querySelector('div.sc-bwzfXH.h3us20-0.cNYGOs').children];
      const opcionaisList = opcionais.map(o => {
        return o.children[0].children[1].innerText;
      });

      return { title, price, caracteristicas: caracteristicasList, opcionais: opcionaisList, link: '' }
    });

    const existe = await Carro.findOne({ title: dadosAnuncio.title });
    if (!existe) {
      dadosAnuncio.link = link;
      await Carro.create(dadosAnuncio);
    }

    await browser.close();
    return dadosAnuncio;
  } catch (error) {
    await browser.close();
    console.log(link);
    return error.message;
  }
}