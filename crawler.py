from selenium import webdriver
from bs4 import BeautifulSoup

# def crawl_print_address(url):
#     """
#     Crawls the given URL and prints all URLs starting with '/en/players'
#     found in the HTML.
#
#     Args:
#         url (str): The URL to crawl.
#
#     Returns:
#         None
#     """
#     # Initialize the WebDriver
#     driver = webdriver.Chrome()
#     driver.get(url)
#
#     # Wait for the page to load completely
#     driver.implicitly_wait(10)
#
#     # Get the page source after JavaScript execution
#     html = driver.page_source
#
#     # Parse the HTML content
#     soup = BeautifulSoup(html, 'html.parser')
#
#     # Find all links starting with '/en/players/'
#     for link in soup.find_all('a', href=True):
#         href = link.get('href')
#         if href.startswith('/en/players/'):
#             print(href)
#
#     # Close the WebDriver
#     driver.quit()
#
# # crawl_print_address('https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats')
# print()

def crawl_save_address(url, filename):
    """
     Crawls the given URL, extracts addresses from the HTML content, and saves them to a file.

    Args:
        url (str): The URL to crawl.
        filename (str): The name of the file to save the addresses to.

    Returns:
        None
    """
    # Initialize the WebDriver
    driver = webdriver.Chrome()

    # Load the webpage
    driver.get(url)

    # Wait for the page to load completely
    driver.implicitly_wait(10)

    # Get the page source after JavaScript execution
    html = driver.page_source

    # Parse the HTML content
    soup = BeautifulSoup(html, 'html.parser')

    # Save all links starting with '/en/players/' in a text file
    with open(filename, 'w') as file:
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href.startswith('/en/players/'):
                file.write('https://fbref.com' + href + '\n')

    # Close the WebDriver
    driver.quit()

crawl_save_address('https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats', 'players.txt')


# Clean the list
def delete_rows(filename, num_rows):
    # Read the content of the file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Skip the first 'num_rows' lines
    new_lines = lines[num_rows:]

    # Write the remaining content back to the file
    with open(filename, 'w') as file:
        file.writelines(new_lines)

delete_rows("players.txt", 53)


def delete_last_rows(filename, num_rows):
    # Read the content of the file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Remove the last 'num_rows' lines
    new_lines = lines[:-num_rows]

    # Write the remaining content back to the file
    with open(filename, 'w') as file:
        file.writelines(new_lines)

delete_last_rows("players.txt", 17)


def write_odd_rows_to_new_file(input_filename, output_filename):
    # Read the content of the input file
    with open(input_filename, 'r') as input_file:
        lines = input_file.readlines()

    # Select only the odd-indexed rows
    odd_rows = [line for index, line in enumerate(lines) if index % 2 == 0]

    # Write the odd rows to the new file
    with open(output_filename, 'w') as output_file:
        output_file.writelines(odd_rows)

write_odd_rows_to_new_file("players.txt", "all_players.txt")

import os
os.remove('players.txt')

### remove duplicate links

def remove_duplicates_preserving_order(file_name):
    with open(file_name, 'r') as file:
        urls = file.readlines()

    unique_urls = []
    seen = set()
    for url in urls:
        if url not in seen:
            unique_urls.append(url)
            seen.add(url)

    with open(file_name, 'w') as file:
        file.writelines(unique_urls)

file_name = 'all_players_URL.txt'
remove_duplicates_preserving_order(file_name)

print('Done. File saved. Duplicates removed')


