import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import os

directory = 'data/HTMLs/'
data = []

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)

    # Open and read the HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        html = file.read()

        soup = BeautifulSoup(html, 'html.parser')

        table = soup.find('table', id='table')

        for row in table.find_all('tr')[0:]:  # Skip the first two header rows
            cells = row.find_all('td')
            if cells:
                verified_icon = cells[1].find('img', class_='table-verification')
                verified = verified_icon and verified_icon[
                    'src'] == "https://capology-e6a3.kxcdn.com/static/images/icons/verified-green.svg"

                data.append({
                    "name": cells[0].text.strip(),
                    "verified": verified,
                    "weekly_gross_eur": int(cells[2].text.strip().replace('€', '').replace(',', '').replace(' ', '')) if
                    cells[2].text.strip().replace('€', '').replace(',', '').replace(' ', '').isdigit() else None,
                    # Remove spaces
                    "annual_gross_eur": int(cells[3].text.strip().replace('€', '').replace(',', '').replace(' ', '')) if
                    cells[3].text.strip().replace('€', '').replace(',', '').replace(' ', '').isdigit() else None,
                    # Remove spaces
                    "bonus_gross_eur": int(cells[4].text.strip().replace('€', '').replace(',', '').replace(' ', '')) if
                    cells[4].text.strip().replace('€', '').replace(',', '').replace(' ', '').isdigit() else None,
                    "signed": cells[5].text.strip(),
                    "expiration": cells[6].text.strip(),
                    "years": int(cells[7].text.strip()) if cells[7].text.strip().isdigit() else None,
                    "total_gross_eur": int(cells[8].text.strip().replace('€', '').replace(',', '').replace(' ', '')) if
                    cells[8].text.strip().replace('€', '').replace(',', '').replace(' ', '').isdigit() else None,
                    "release_clause": int(cells[9].text.strip().replace('€', '').replace(',', '').replace(' ', '')) if
                    cells[9].text.strip().replace('€', '').replace(',', '').replace(' ', '').isdigit() else None,
                    "status": cells[10].text.strip(),
                    "pos": cells[11].text.strip(),
                    "pos_detail": cells[12].text.strip(),
                    "age": int(cells[13].text.strip()) if cells[13].text.strip().isdigit() else None,
                    "country": cells[14].text.strip(),
                    "club": cells[15].text.strip(),
                })





df = pd.DataFrame(data)

df.to_csv("data/capology_data.csv", index=False)
print("done")