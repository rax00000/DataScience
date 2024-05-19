# v4
# acelasi cod doar restrcturat
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time

columns = ["Player Name",
           "Position",
           "Club",
           "Annual wage in EUR",
           "Non-Penalty Goals",
           "Shots Total",
           "Assists",
           "Shot-Creating Actions",
           "Passes Attempted",
           "Pass Completion %",
           "Progressive Passes",
           "Progressive Carries",
           "Successful Take-Ons",
           "Touches (Att Pen)",
           "Progressive Passes Rec",
           "Tackles",
           "Interceptions",
           "Blocks",
           "Clearances",
           "Aerials Won",
           "Goals Against",
           "Save Percentage",
           "Save% (Penalty Kicks)",
           "Clean Sheet Percentage",
           "Touches",
           "Launch %",
           "Goal Kicks",
           "Avg. Length of Goal Kicks",
           "Crosses Stopped %",
           "Def. Actions Outside Pen. Area",
           "Avg. Distance of Def. Actions"
           ]

big_5_all_player_data_fbref = pd.DataFrame(columns=columns)

filename = 'C:/Users/Radu Leonte/Desktop/MSc Courses/Data Science Project/scraper 1/all_players.txt'

with open(filename, 'r') as file:
    urls = file.readlines()

def extract_all_player_data(soup):
    # Initialize a dictionary to store all player data
    player_data = {
        "Player Name": "",
        "Position": "",
        "Club": "",
        "Annual wage in EUR": "",
        "Non-Penalty Goals": "",
        "Shots Total": "",
        "Assists": "",
        "Shot-Creating Actions": "",
        "Passes Attempted": "",
        "Pass Completion %": "",
        "Progressive Passes": "",
        "Progressive Carries": "",
        "Successful Take-Ons": "",
        "Touches (Att Pen)": "",
        "Progressive Passes Rec": "",
        "Tackles": "",
        "Interceptions": "",
        "Blocks": "",
        "Clearances": "",
        "Aerials Won": "",
        "Goals Against": "",
        "Save Percentage": "",
        "Save% (Penalty Kicks)": "",
        "Clean Sheet Percentage": "",
        "Touches": "",
        "Launch %": "",
        "Goal Kicks": "",
        "Avg. Length of Goal Kicks": "",
        "Crosses Stopped %": "",
        "Def. Actions Outside Pen. Area": "",
        "Avg. Distance of Def. Actions": ""
    }

    # Extract Player Name
    title_tag = soup.find('title')
    if title_tag:
        player_name = title_tag.text.split('|')[0].strip()
        player_name = player_name.split("Stats")[0].strip()
        player_data["Player Name"] = player_name

    # Extract meta information
    meta_info = soup.find('div', {'id': 'meta'})
    if meta_info:
        # Extract Position
        position_p = meta_info.find(lambda tag: tag.name == 'p' and 'Position:' in tag.text)
        if position_p:
            position = position_p.find('strong', text='Position:').next_sibling.strip()
            position = ''.join(filter(str.isalnum, position))
            # print(position)
            player_data["Position"] = position.replace('\xa0', ' ')  # Replace non-breaking spaces

        # Extract Club
        club_p = meta_info.find(lambda tag: tag.name == 'p' and 'Club:' in tag.text)
        if club_p and club_p.find('a'):
            club_name = club_p.find('a').text.strip()
            player_data["Club"] = club_name

        # Extract Annual Wages in Euros
        wages_p = meta_info.find(lambda tag: tag.name == 'p' and 'Wages' in tag.text)

        # print(wages_p)

        if wages_p:
            wage_span = wages_p.find('span', {'data-tip': True})
            if wage_span:
                data_tip = wage_span['data-tip']

                # print("print(data tip)")
                # print(data_tip)
                # Use BeautifulSoup to parse the HTML-like content within data-tip
                soup_tip = BeautifulSoup(data_tip, 'html.parser')
                # Find the text that contains "Wages in Euros"
                # print("Parsed content:", soup_tip.prettify())

                euros_text = soup_tip.find(text=re.compile("€"))
                # print(euros_text)
                if euros_text:
                    # print("Found 'Wages in Euros' text:", euros_text)
                    # Extract the next sibling text which should contain the annual wage in euros
                    annual_euro_wage = euros_text.find_next(text=re.compile("Annual:"))
                    if annual_euro_wage:
                        euro_amount = re.search(r"Annual: € ([\d,]+)", euros_text)
                        if euro_amount:
                            annual_euro_wage = euro_amount.group(1).replace(',', '')  # Remove commas for consistency
                            player_data["Annual wage in EUR"] = annual_euro_wage
                            # print("Extracted Annual Wage in EUR:",
                            #       annual_euro_wage)  # Debugging line to view extracted wage

    # Find the scouting report table
    scouting_report_table = soup.find('div', {'id': 'all_scout_summary'})
    if scouting_report_table:
        rows = scouting_report_table.find_all('tr')
        for row in rows:
            header = row.find('th')
            if header:
                header_text = header.text.strip()
                data_cell = row.find('td')
                if data_cell and header_text in player_data:
                    player_data[header_text] = data_cell.text.strip()

    return player_data

x=1

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',  # Do Not Track Request Header
    'Connection': 'keep-alive'
}
x=0

for url in urls:
    success = False
    while not success:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            player_data = extract_all_player_data(soup)
            new_row = pd.DataFrame([player_data])
            big_5_all_player_data_fbref = pd.concat([big_5_all_player_data_fbref, new_row], ignore_index=True)
            success = True  # Set success to True to exit the loop

            print(x) # counter to track progress
            x=x+1
            time.sleep(4)

        elif response.status_code == 429:
            retry_after = response.headers.get('Retry-After', 30)  # Use the Retry-After header if available, otherwise default to 60 seconds
            print(f"Rate limit exceeded, retrying in {retry_after} seconds...")
            time.sleep(int(retry_after))  # Wait before retrying
        else:
            print(f"Failed to retrieve data from {url}, status code {response.status_code}")
            time.sleep(5)
            break  # Exit loop on other errors

big_5_all_player_data_fbref.to_csv('C:/Users/Radu Leonte/Desktop/MSc Courses/Data Science Project/scraper 1/big_5_all_player_data.csv', index=False)
big_5_all_player_data_fbref.to_excel('C:/Users/Radu Leonte/Desktop/MSc Courses/Data Science Project/scraper 1/big_5_all_player_data.xlsx', index=False)

print("Done")

