import time
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import openpyxl

big_table = pd.read_csv('all_players_scouting_report.csv')
small_table = pd.read_csv('player_stats_small_table_clean.csv')

df_merged = pd.merge(big_table, small_table, how='inner', on= ['Player Name'])
df_merged.to_csv('merged_data_for_analysis.csv')
print('Done')