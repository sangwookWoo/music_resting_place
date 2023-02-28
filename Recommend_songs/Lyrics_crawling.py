#Step 1. 필요한 모듈을 로딩합니다
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.select import Select
from selenium.webdriver.edge.options import Options
from bs4 import BeautifulSoup
import time          


driver = webdriver.Edge('C:\develop_dir\playdata_project\project3\edgedriver_win64\msedgedriver.exe') # 개인에게 맞는 경로 수정 필요
song_list = []
lyrics_archive = []
with open("playdata_project/project3/song_list.txt", "r", encoding='utf-8') as f:
    for line in f:
        song_list.append(line.strip()) # 저장해둔 노래 리스트에서 한 곡씩 요소화
for i in song_list:
    driver.get(url='https://www.melon.com/')
    query = driver.find_element(By.CLASS_NAME, "ui-autocomplete-input") # 검색창
    query.send_keys(f'{i}') # 리스트에서 가져온 한 곡을 타이핑
    query.send_keys(Keys.RETURN) # 검색
    time.sleep(1)
    driver.execute_script('window.scrollTo(0, 500)')
    driver.find_element(By.XPATH, '//*[@id="conts"]/div[5]/div/ul/li/dl/dd[1]/a').click() # 가사 란 클릭
    driver.execute_script('window.scrollTo(0, 500)')
    time.sleep(1)
    driver.find_element(By.XPATH, '//*[@id="lyricArea"]/button').click()
    time.sleep(1)
    lyrics = driver.find_element(By.ID, 'd_video_summary')
    time.sleep(1)
    get_lyrics = lyrics.text
    lyrics_archive.append(get_lyrics) # 일단 여기까지 만듬
driver.close()