# import schedule
import requests
from bs4 import BeautifulSoup
from time import sleep, time
from lxml import etree
import json
import pandas as pd
from threading import Thread


class Th(Thread):
    def __init__(self, links):
        Thread.__init__(self)
        self.links = links

    def run(self):
        pass


def coletaLinksPage(page, linksExistentes):
    header = {
        'Host': 'se.olx.com.br',
        'Connection': 'close',
        'Cache-Control': 'max-age=0',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-User': '?1',
        'Sec-Fetch-Dest': 'document',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7'
    }

    result = requests.get(
        "https://se.olx.com.br/autos-e-pecas/carros-vans-e-utilitarios?o="+str(page), headers=header)
    html = result.content
    soup = BeautifulSoup(html, 'html.parser')

    itens = soup.find_all(id="ad-list")

    links = []

    for item in itens:
        carros = item.find_all('a')
        for carro in carros:
            link = carro.get('href')
            if not linksExistentes.query(f'links=="{link}"').shape[0]:
                links.append(link)

    # print(page)

    return links


def coletaLinks():
    inicio = time()
    totalLinks = []
    linksExistentes = pd.read_csv('links.csv')
    print(f"linksExistentes {len(linksExistentes)}")
    for i in range(1, 101):
        links = coletaLinksPage(i, linksExistentes)
        for f in links:
            totalLinks.append(f)

    pd.concat([linksExistentes, pd.DataFrame(totalLinks, columns=['links'])]
              ).to_csv('links.csv', index=False)

    print(f"Total de links a coletados: {len(totalLinks)}")

    fim = time()

    print(f"Tempo para coletar os links {round(fim-inicio, 2)} s")

    return totalLinks


def coletaDados(link):
    header = {
        'Host': 'se.olx.com.br',
        'Connection': 'close',
        'Cache-Control': 'max-age=0',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-User': '?1',
        'Sec-Fetch-Dest': 'document',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7'
    }
    result = requests.get(link, headers=header)
    html = result.content
    soup = BeautifulSoup(html, 'html.parser')

    dom = etree.HTML(str(soup))

    preco = dom.xpath(
        '/html/body/div[2]/div/div[4]/div[2]/div/div[2]/div[2]/div[17]/div/div/div[1]/div[2]/h2[2]')[0].text
    title = dom.xpath(
        '/html/body/div[2]/div/div[4]/div[2]/div/div[2]/div[1]/div[17]/div/div/div/div/h1')[0].text

    opicionaisList = [span.text for span in soup.findAll(
        'span', class_="sc-1g2w54p-0 bCoOvQ sc-ifAKCX cmFKIN")]

    opicionais = {"Vidro elétrico": 0, "Trava elétrica": 0, "Ar condicionado": 0, "Direção hidráulica": 0,
                  "Som": 0, "Air bag": 0, "Alarme": 0, "Sensor de ré": 0, "Câmera de ré": 0, "Blindado": 0}

    for o in opicionaisList:
        if o in opicionais:
            opicionais[o] = 1

    divCaracteristicas = soup.findAll(
        'div', class_="sc-hmzhuo HlNae sc-jTzLTM iwtnNi")

    caracteristicas = {}
    for caracteristica in divCaracteristicas:
        cTitle = caracteristica.find('span', class_='sc-ifAKCX dCObfG').text
        cDesc = None
        if caracteristica.find('a') is None:
            cDesc = caracteristica.find('span', class_='sc-ifAKCX cmFKIN').text
        else:
            cDesc = caracteristica.find('a').text
        caracteristicas[cTitle] = cDesc

    return {
        'title': title,
        'price': preco,
        **caracteristicas,
        **opicionais,
        'link': link
    }


def main():
    links = coletaLinks()
    inicio = time()

    unattended = open('unattended.csv', 'w')
    arq = open('carros.csv', 'w')
    arq.write('Modelo,Marca,Tipo de veículo,Ano,Quilometragem,Potência do motor,Combustível,Câmbio,Direção,Cor,Portas,Final de placa,Vidro elétrico,Trava elétrica,Ar condicionado,Direção hidráulica,Som,Air bag,Alarme,Sensor de ré,Câmera de ré,Blindado,Valor\n')

    error = 0

    for l in links:
        try:
            dados = coletaDados(l)
            unattended.write(f"{json.dumps(dados)}\n")
            arq.close()
            arq = open('carros.csv', 'a')
            arq.write(f'{dados["Modelo"]},{dados["Marca"]},{dados["Tipo de veículo"]},{dados["Ano"]},{dados["Quilometragem"]},{dados["Potência do motor"]},{dados["Combustível"]},{dados["Câmbio"]},{dados["Direção"]},{dados["Cor"]},{dados["Final de placa"]},{dados["Vidro elétrico"]},{dados["Trava elétrica"]},{dados["Ar condicionado"]},{dados["Direção hidráulica"]},{dados["Som"]},{dados["Air bag"]},{dados["Alarme"]},{dados["Sensor de ré"]},{dados["Câmera de ré"]},{dados["Blindado"]},{dados["price"]}\n')
        except:
            error = error + 1
            # print(f"Erro na coleda desse dado - {l}")

    print(f"Quantidade de Erros: {error}")
    fim = time()
    print(f"Tempo para coletar os dados {round(fim-inicio, 2)} s")
    print(
        f"Tempo médio para coletar os dados {round(fim-inicio, 2)/len(links)} s")


# schedule.every().hour.do(main)
# while 1:
#     schedule.run_pending()
#     sleep(1)

main()
