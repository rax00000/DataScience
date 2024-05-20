import time
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd


def scrape_player_stats(url):
    # Initialize the WebDriver
    driver = webdriver.Chrome()

    # Load the webpage
    driver.get(url)

    # Wait for the table to load
    time.sleep(5)

    # Get the page source after JavaScript execution
    html = driver.page_source

    # Parse the HTML content
    soup = BeautifulSoup(html, 'html.parser')

    # Find the table with player standard stats
    table = soup.find('table', {'id': 'stats_standard'})

    # Extract specific information from the table
    if table:
        # Create an empty list to store data
        data = []

        # Get the rows of the table
        rows = table.find_all('tr')

        # Iterate over rows, skipping the header row
        for row in rows[1:]:
            # Extract data from each row
            cells = row.find_all('td')
            # Check if the number of cells is as expected
            if len(cells) >= 3:  # Assuming there are at least 3 cells for player name, nationality, and position
                player_name = cells[0].text.strip()
                nationality_full = cells[1].text.strip()
                nationality = nationality_full[-3:]  # Extract the last 3 characters
                position = cells[2].text.strip()
                league = cells[4].text.strip()
                age_full = cells[5].text.strip()
                age = age_full[:2]  # Extract the first 2 characters
                matches = cells[7].text.strip()
                starts = cells[8].text.strip()
                minutes = cells[9].text.strip()
                goals = cells[11].text.strip()
                assists = cells[12].text.strip()
                penalty_made = cells[15].text.strip()
                penalty_attempt = cells[16].text.strip()

                # Append the extracted data to the list
                data.append([player_name, nationality, position, league, age, matches, starts, minutes, goals, assists,
                             penalty_made, penalty_attempt])

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data, columns=['Player Name', 'Nationality', 'Position', 'League', 'Age', 'Matches', 'Starts',
                                         'Minutes', 'Goals', 'Assists', 'Penalty_made', 'Penalty_attempted'])

        # Close the WebDriver
        driver.quit()

        return df
    else:
        print("Table with player standard stats not found on the webpage.")
        # Close the WebDriver
        driver.quit()
        return None


# URL of the webpage to scrape
url = "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats"

# Call the function to scrape player stats and create a DataFrame
player_stats_df = scrape_player_stats(url)

# Remove the country code of the league from the 'League' column of the DataFrame
league_mapping = {'eng Premier League': 'Premier League',
                  'de Bundesliga': 'Bundesliga',
                  'fr Ligue 1': 'Ligue 1',
                  'es La Liga': 'La Liga',
                  'it Serie A': 'Serie A'}

player_stats_df['League'] = player_stats_df['League'].replace(league_mapping)

# Print the DataFrame
player_stats_df.shape

player_stats_df.loc[(player_stats_df['Player Name'] == 'Vitinha') & (player_stats_df['Age'] == 24) & (player_stats_df['League'] == 'Serie A'), 'Player Name'] = 'Vitinha_1'
player_stats_df.loc[(player_stats_df['Player Name'] == 'Vitinha') & (player_stats_df['Age'] == 24) & (player_stats_df['League'] == 'Ligue 1'), 'Player Name'] = 'Vitinha_2'
player_stats_df.loc[(player_stats_df['Player Name'] == 'Danilo') & (player_stats_df['Age'] == 23), 'Player Name'] = 'Danilo_1'
player_stats_df.loc[(player_stats_df['Player Name'] == 'Danilo') & (player_stats_df['Age'] == 32), 'Player Name'] = 'Danilo_2'
player_stats_df.loc[(player_stats_df['Player Name'] == 'Fernando') & (player_stats_df['Age'] == 33), 'Player Name'] = 'Fernando_1'
player_stats_df.loc[(player_stats_df['Player Name'] == 'Fernando') & (player_stats_df['Age'] == 36), 'Player Name'] = 'Fernando_2'
player_stats_df.loc[(player_stats_df['Player Name'] == 'Mamadou Coulibaly') & (player_stats_df['Age'] == 25), 'Player Name'] = 'Mamadou Coulibaly_1'
player_stats_df.loc[(player_stats_df['Player Name'] == 'Mamadou Coulibaly') & (player_stats_df['Age'] == 20), 'Player Name'] = 'Mamadou Coulibaly_2'
player_stats_df.loc[(player_stats_df['Player Name'] == 'Marquinhos') & (player_stats_df['Age'] == 30), 'Player Name'] = 'Marquinhos_1'
player_stats_df.loc[(player_stats_df['Player Name'] == 'Marquinhos') & (player_stats_df['Age'] == 21), 'Player Name'] = 'Marquinhos_2'
player_stats_df.loc[(player_stats_df['Player Name'] == 'Rodri') & (player_stats_df['Age'] == 27), 'Player Name'] = 'Rodri_1'
player_stats_df.loc[(player_stats_df['Player Name'] == 'Rodri') & (player_stats_df['Age'] == 24), 'Player Name'] = 'Rodri_2'
player_stats_df.loc[(player_stats_df['Player Name'] == 'Stefan Mitrović') & (player_stats_df['Age'] == 21), 'Player Name'] = 'Stefan Mitrović_1'
player_stats_df.loc[(player_stats_df['Player Name'] == 'Stefan Mitrović') & (player_stats_df['Age'] == 33), 'Player Name'] = 'Stefan Mitrović_2'
player_stats_df.loc[(player_stats_df['Player Name'] == 'Juan Cruz') & (player_stats_df['League'] == 'Serie A'), 'Player Name'] = 'Juan Cruz_1'
player_stats_df.loc[(player_stats_df['Player Name'] == 'Juan Cruz') & (player_stats_df['League'] == 'La Liga'), 'Player Name'] = 'Juan Cruz_2'

player_stats_df['Minutes'] = player_stats_df['Minutes'].str.replace(',', '').astype(int)

# Sort the DataFrame by 'Player Name' and 'Minutes' in descending order so the max minutes row comes first
player_stats_df = player_stats_df.sort_values(by=['Player Name', 'Minutes'], ascending=[True, False])

# Aggregate data: keep the first entry for columns like 'Age', 'Position', 'League' where the max 'Minutes' row is wanted
# and sum the numerical stats like 'Minutes', 'Starts', 'Goals', etc.

aggregated_df = player_stats_df.groupby('Player Name').agg({
    'Nationality': 'first',
    'Position': 'first',
    'League': 'first',
    'Age': 'first',
    'Matches': 'sum',
    'Starts': 'sum',
    'Minutes': 'sum',
    'Goals': 'sum',
    'Assists': 'sum',
    'Penalty_made': 'sum',
    'Penalty_attempted': 'sum'
}).reset_index()


print(aggregated_df['Player Name'].is_unique)

aggregated_df.to_csv('player_stats_small_table_clean.csv', index = False)
print('Done')
