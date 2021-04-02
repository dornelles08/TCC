const puppeteer = require("puppeteer");
const mongoose = require("mongoose");
const cron = require("node-cron");

const Anuncios = require("../model/Anuncios");
const Carro = require("../model/Carro");

async function coletaLinks() {
  await mongoose.connect(
    "mongodb+srv://geral:geral@cluster0.sg3qs.mongodb.net/TCC?retryWrites=true&w=majority",
    {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    }
  );

  const browser = await puppeteer.launch({
    headless: false,
    args: ["--no-sandbox"],
  });
  const page = await browser.newPage();

  for (let i = 1; i <= 100; i++) {
    try {
      await page.goto(
        `https://se.olx.com.br/autos-e-pecas/carros-vans-e-utilitarios?o=${i}`
        // `https://sp.olx.com.br/sao-paulo-e-regiao/autos-e-pecas/carros-vans-e-utilitarios?o=${i}`
      );

      const linkAnuncios = await page.evaluate(() => {
        const nodeList = document.querySelector("#ad-list");
        const itens = nodeList.querySelectorAll("a");
        const carrosArray = [...itens];

        const linksAnuncios = carrosArray.map((anuncio) => {
          return {
            link: anuncio.getAttribute("href"),
          };
        });
        return linksAnuncios;
      });

      linkAnuncios.forEach(async (link) => {
        const existe = await Carro.findOne({ link: link.link });
        const existeAnuncio = await Anuncios.findOne({ link: link.link });
        if (!existe && !existeAnuncio) {
          await Anuncios.create(link);
        }
      });
      console.log(`Pagina ${i}`);
    } catch (error) {
      console.log(error.message);
    }
  }

  await browser.close();

  console.log("Finalizado");
}

coletaLinks();
cron.schedule("0 * * * *", coletaLinks);
