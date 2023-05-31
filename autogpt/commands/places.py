import os
import json
import copy
import time
import concurrent.futures
from tqdm import tqdm

import tiktoken
import openai
from trafilatura import fetch_url, extract
from duckduckgo_search import DDGS
from tenacity import retry, stop_after_attempt, wait_random_exponential

from selenium import webdriver
from pathlib import Path
from webdriver_manager.chrome import ChromeDriverManager
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from haversine import haversine
import yaml

import numpy as np

import sys
sys.path.append('.')
from autogpt.commands.command import command
from autogpt.config import Config
from autogpt.llm.llm_utils import create_chat_completion

CFG = Config()
if CFG.workspace_path is None:
    CFG.workspace_path = Path.cwd()

def shorten_url(url):
    apiurl = f"http://tinyurl.com/api-create.php?url={url}"
    response = requests.get(apiurl)
    return response.text

def get_selenium_driver():
    chromium_driver_path = Path("/usr/bin/chromedriver")
    options = webdriver.ChromeOptions()
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36"
    )
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(
        executable_path=chromium_driver_path
        if chromium_driver_path.exists()
        else ChromeDriverManager().install(),
        options=options,
    )
    return driver

def get_summarized_text(place, extra_question=''):
    place['extra_questoin'] = extra_question
    prompt = f""" {place}

Please output using the following format in English:
{{
    "description_reviews_rating_summary_in_brief_polite": "<summary>",
    "answer_to_extra_question_if_exists": "<answer>",
    "extra_question_info_exists": "<1 or 0>"
}}
"""
    del place['extra_questoin']
    #print(prompt)
    response = create_chat_completion(messages=[{'role': 'user', 'content': prompt}], model=CFG.fast_llm_model, temperature=0.5)
    return response


def get_reviews(soup, n_res=10):
    reviews_list = []
    
    reviews = soup.find_all('li', {'class': 'YeINN'})
    
    for i, review in enumerate(reviews[:n_res]):
        # 리뷰 텍스트 추출
        try:
            review_text = review.find('span', {'class': 'zPfVt'}).text
            review_text = " ".join(review_text.split())
            reviews_list.append(review_text)
        except:
            pass
    
    return reviews_list


def get_place_details(place_url, extra_question=''):
    print(f'BROWSING: {place_url}')
    driver = get_selenium_driver()
    details = {
        'name': None,
        'url': place_url,
        'type': None,
        'phone': None,
        'rating': None,
        'n_visitor_reviews': None,
        'n_blog_reviews': None,
        'reviews': None,
        'address': None,
        'operating_hours': None,        
        #'street_view_url': None,
        #'price': None
    }
    
    driver.get(place_url)
    time.sleep(2)

    # HTML 추출
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # 기존 정보 추출
    details['name'] = soup.select_one('div#_title span.Fc1rA').text if soup.select_one('div#_title span.Fc1rA') else None
    details['type'] = soup.select_one('div#_title span.DJJvD').text if soup.select_one('div#_title span.DJJvD') else None
    details['rating'] = soup.select_one('span.PXMot.LXIwF em').text if soup.select_one('span.PXMot.LXIwF em') else None
    details['n_visitor_reviews'] = soup.select_one('span.PXMot a[href*="review/visitor"] em').text if soup.select_one('span.PXMot a[href*="review/visitor"] em') else None
    details['n_blog_reviews'] = soup.select_one('span.PXMot a[href*="review/ugc"] em').text if soup.select_one('span.PXMot a[href*="review/ugc"] em') else None

    # 추가 정보 추출
    details['address'] = soup.select_one('div.O8qbU.tQY7D div.vV_z_ a.PkgBl span.LDgIH').text if soup.select_one('div.O8qbU.tQY7D div.vV_z_ a.PkgBl span.LDgIH') else None
    
    # 운영시간 추출    
    #details['operating_hours'] = soup.select_one('.O8qbU.pSavy .A_cdD .U7pYf span.place_blind').text if soup.select_one('.O8qbU.pSavy .A_cdD .U7pYf span.place_blind') else None
        
    # street_view_url 추출
    #details['street_view_url'] = soup.select_one('span.S8peq a[href^="https://app.map.naver.com/panorama/"]')['href'] if soup.select_one('span.S8peq a[href^="https://app.map.naver.com/panorama/"]') else None

    # 편의 정보 추출
    convenience_elements = soup.select('div.O8qbU div.vV_z_')
    def has_only_text(element):
        if element.find() is None and element.string is not None:
            return True
        return False
    details['convenience'] = "\n".join([element.text for element in convenience_elements if has_only_text(element)])
        
    # 전화번호 추출
    details['phone'] = soup.select_one('span.xlx7Q').text if soup.select_one('span.xlx7Q') else None    
    
    # 가격표 추출
    #price_list = soup.select('div.JLkY7')
    #if price_list:
    #    details['price'] = []
    #    for item in price_list:
    #        price_item = item.select_one('span.A_cdD').text
    #        price_value = item.select_one('div.CLSES').text
    #        details['price'].append({price_item: price_value})
        
    # Description 추출
    description_section = soup.select_one('div.O8qbU.dRAr1 div.vV_z_ a.xHaT3 span.zPfVt')
    if description_section:
        details['description'] = " ".join(description_section.text.split())

    # reviews
    driver.get(f'{place_url}/review/visitor')
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    details['reviews'] = get_reviews(soup)
    
    # photo 추출
    driver.get(f'{place_url}/photo')
    time.sleep(2)
    if 'photo' in driver.current_url:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        photo = soup.find('div', {'class': 'place_section_content'}).find('img')        
        photo = shorten_url(photo['src'])
        details['photo'] = photo
        
    # summarize
    try:
        summarized_output = get_summarized_text(details, extra_question)
        summarized_output = eval(summarized_output)
        details['summary'] = list(summarized_output.values())[0]
        details['extra_question'] = extra_question
        details['answer_to_extra_question'] = list(summarized_output.values())[1]
        details['extra_question_info_exists'] = list(summarized_output.values())[2]
    except:
        details['summary'] = ''
        #details['place_score'] = ''
        
    # 많은 토큰 수를 차지하는 reviews와 description을 삭제한다.
    del details['reviews']
    if 'description' in details:
        del details['description']
    
    return details

@command(
    "search_places",
    "Search locations and destinations, returning top_n results, calculate their distance matrix in km, and save to file. "
    "If you have 'search_details', recommend a larger value for 'n_top' than the needed number of places",
    '"search_keyword": "<Examples:강릉 여행지, 제주 가볼만한 곳, 신림 근처, 고기 맛집; avoid ranking numbers; instead use top_n arg>", '
    '"filename": "<yaml_filename>", "top_n": "<default_and_max:7>", "search_details": "<Examples:뷰 좋은 곳, 주차 가능한가요?>"',
)
def search_places(search_keyword, filename, top_n=5, search_details=""):
    top_n = int(top_n)
    driver = get_selenium_driver()
    naver_map_search_url = f'https://m.map.naver.com/search2/search.naver?query={search_keyword}'
    driver.get(naver_map_search_url)
    time.sleep(2)
    elements = driver.find_elements_by_css_selector('li._item._lazyImgContainer')[:top_n]
    if len(elements) == 0:
        driver.quit()
        return f"No results found for {search_keyword}. Try using a simpler search_keyword arg."
    def get_place_details_(place_url):
        return get_place_details(place_url, search_details)
    with concurrent.futures.ThreadPoolExecutor(len(elements)) as executor:
        results = list(executor.map(get_place_details_, [(f"https://m.place.naver.com/place/{element.get_attribute('data-id')}") for element in elements]))    
    #results = []
    for i in range(len(elements)):
        #place_url = f"https://m.place.naver.com/place/{elements[i].get_attribute('data-id')}"
        #place_details = get_place_details(place_url, query)
        results[i]['longitude'] = float(elements[i].get_attribute('data-longitude'))
        results[i]['latitude'] = float(elements[i].get_attribute('data-latitude'))
        
    driver.quit()
    
    num_places = len(results)
    distances = np.zeros((num_places, num_places))
    for i in range(num_places):
        for j in range(num_places):
            loc_i = (results[i]['latitude'], results[i]['longitude'])
            loc_j = (results[j]['latitude'], results[j]['longitude'])
            distances[i, j] = haversine(loc_i, loc_j)

    for result in results:
        del result['longitude'], result['latitude']
    
    distances_str = np.array2string(distances, precision=1, floatmode='fixed')    
    results = {"candidates": results, "distance_matrix": distances_str}

    with open(os.path.join(CFG.workspace_path, filename), 'w') as f:
        yaml.dump(results, f, allow_unicode=True)
    
    # place name 리스트 추출
    place_names = [f"{res['name']}({res['type']})" for res in results['candidates']]
    
    return_msg = (
        f"The details of the found places:{place_names} have been written to {filename}."
        " Information related to search_details (if it exists) has also been written, along with the distance_matrix for all places."
        " If you want to search other types of places, rerun this command with a different keyword."
        #f" Please don't directly read '{filename}'. It could lead to a significant increase in our costs."
    )
    if len(place_names) < top_n:
        return_msg += f' The number of found places is {len(place_names)}, which is less than {top_n}. By retrying this command with a simpler search keyword, you may find more places.'
    return return_msg

if __name__ == '__main__':
    print(search_places('부산 여행지', 'busan_top5.yaml', top_n=5))
    
    #get_place_details('https://m.place.naver.com/place/11555552')
    #get_place_details('https://m.place.naver.com/place/1586994430')
    
    #get_place_details('https://m.place.naver.com/place/37942980')
    #get_place_details('https://m.place.naver.com/place/11658148')
    #get_place_details('https://m.place.naver.com/place/998885728')
    #get_place_details('https://m.place.naver.com/place/11859878')
    