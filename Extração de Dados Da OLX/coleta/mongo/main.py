import schedule
import requests
from bs4 import BeautifulSoup
from time import time, sleep
from lxml import etree


def coletaLinksPage(page):
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
            links.append(carro.get('href'))
    print(page)

    return links


def coletaLinks():
    totalLinks = []
    inicio = time()
    for i in range(1, 100):
        links = coletaLinksPage(i)
        for f in links:
            totalLinks.append(f)

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
        '//*[@id="content"]/div[2]/div/div[2]/div[2]/div[17]/div/div/div[1]/div[2]/h2')[0].text
    title = dom.xpath(
        '//*[@id="content"]/div[2]/div/div[2]/div[1]/div[15]/div/div/div/div/h1')[0].text

    opicionais = [span.text for span in soup.findAll(
        'span', class_="sc-1g2w54p-0 bCoOvQ sc-ifAKCX cmFKIN")]

    divCaracteristicas = soup.findAll(
        'div', class_="sc-hmzhuo HlNae sc-jTzLTM iwtnNi")

    caracteristicas = []
    for caracteristica in divCaracteristicas:
        cTitle = caracteristica.find('span', class_='sc-ifAKCX dCObfG').text
        cDesc = None
        if caracteristica.find('a') is None:
            cDesc = caracteristica.find('span', class_='sc-ifAKCX cmFKIN').text
        else:
            cDesc = caracteristica.find('a').text
        caracteristicas.append({'title': cTitle, 'desc': cDesc})

    return {
        'title': title,
        'price': preco,
        'caracteristicas': caracteristicas,
        'opcionais': opicionais,
        'link': link
    }


def main():
    print(time)
    links = coletaLinks()

    for l in links:
        dados = coletaDados(l)
        requests.post('http://localhost:1997/saveOnMongo', json=dados)


schedule.every().hour.do(main)
while 1:
    schedule.run_pending()
    sleep(1)
