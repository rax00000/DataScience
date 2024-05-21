import time
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import openpyxl

big_table = pd.read_csv('data/all_players_scouting_report.csv')
small_table = pd.read_csv('data/player_stats_small_table_clean.csv')

df_merged = pd.merge(small_table, big_table, how='inner', on= ['Player Name'])
df_merged.to_csv('data/merged_data_for_analysis.csv')
print('Done')