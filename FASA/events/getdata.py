from bs4 import BeautifulSoup
import requests
import pandas as pd

url = 'https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/'
ext = 'gz'

def listdb(url, ext=''):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

bigdf = pd.DataFrame()
for file in listdb(url, ext):
    print(file)
    data = pd.read_csv(file, compression='gzip', error_bad_lines=False)
    bigdf = pd.concat([bigdf,data],ignore_index=True)
bigdf.to_csv('events_datasets.csv', index=False)
