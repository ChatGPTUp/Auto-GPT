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

load_dotenv('../.env')
openai.api_key = os.environ.get("OPENAI_API_KEY")


MODELS_INFO = {
    'gpt-3.5-turbo': {'max_tokens': 4096, 'pricing': 0.002/1000, 'tokenizer': tiktoken.get_encoding("cl100k_base"), 'tokens_per_message': 5},
    'gpt-4': {'max_tokens': 4096, 'pricing': 0.03/1000, 'tokenizer': tiktoken.get_encoding("cl100k_base"), 'tokens_per_message': 5},
}

def shorten_url(url):
    apiurl = f"http://tinyurl.com/api-create.php?url={url}"
    response = requests.get(apiurl)
    return response.text

def split_text(text, max_tokens=500, overlap=0, model='gpt-3.5-turbo'):
    tokenizer = MODELS_INFO[model]['tokenizer']
    tokens = tokenizer.encode(text)
    sid = 0
    splitted = []
    while True:
        if sid + overlap >= len(tokens):
            break
        eid = min(sid+max_tokens, len(tokens))
        splitted.append(tokenizer.decode(tokens[sid:eid]))
        sid = eid - overlap
    return splitted

def truncate_messages(messages, system_prompt="", model='gpt-3.5-turbo', n_response_tokens=500, keep_last=False):
    max_tokens = MODELS_INFO[model]['max_tokens']
    n_tokens_per_message = MODELS_INFO[model]['tokens_per_message']
    tokenizer = MODELS_INFO[model]['tokenizer']
    n_used_tokens = 3 + n_response_tokens
    n_used_tokens += n_tokens_per_message + len(tokenizer.encode(system_prompt))
    iterator = range(len(messages))
    if keep_last: 
        iterator = reversed(iterator)
    for i in iterator:
        message = messages[i]
        n_used_tokens += n_tokens_per_message
        if n_used_tokens >= max_tokens:
            messages = messages[i+1:] if keep_last else messages[:i]
            print('Messages Truncated')
            break
        content_tokens = tokenizer.encode(message['content'])
        n_content_tokens = len(content_tokens)
        n_used_tokens += n_content_tokens
        if n_used_tokens >= max_tokens:
            truncated_content_tokens = content_tokens[n_used_tokens-max_tokens:] if keep_last else content_tokens[:max_tokens-n_used_tokens]
            other_messages = messages[i+1:] if keep_last else messages[:i]
            messages = [{'role': message['role'], 'content': tokenizer.decode(truncated_content_tokens)}] + other_messages
            print('Messages Truncated')
            break
    return messages


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def get_chatgpt_response(messages:list, system_prompt="", model='gpt-3.5-turbo', temperature=0.5, keep_last=True):
    messages = copy.deepcopy(messages)
    messages = truncate_messages(messages, system_prompt, model, keep_last=keep_last)
    messages = [{"role": "system", "content": system_prompt}]+messages
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    response = dict(completion.choices[0].message)
    response['dollars_spent'] = completion['usage']['total_tokens'] * MODELS_INFO[model]['pricing']
    return response

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

def create_message(chunk: str, question: str):
    if question != '':
        content = f'"""{chunk}""" Using the above text, answer the following'
        f' question: "{question}" -- if the question cannot be answered using the text,'
        " summarize the text. Please output in the language used in the above text.",
    else:
        content = (
            f'"""{chunk}"""'
            '\nSummarize above reviews.'
        )
    
    return {
        "role": "user",
        "content": content
    }

def summarize_reviews(reviews, query):
    message = create_message("\n".join(reviews), query)
    response = get_chatgpt_response([message], model='gpt-3.5-turbo', temperature=0)['content']
    return response


def summarize_info(info):
    if not info:
        return ''
    prompt = f"""```
{info}
```
Above text is extracted from html info of a place. Concisely extract key informations from it.
Key Information:
"""
    response = get_chatgpt_response([{'role': 'user', 'content': prompt}], model='gpt-3.5-turbo', temperature=0)['content']
    return response

def get_summarized_text(place, extra_question=''):
    place['extra_questoin'] = extra_question
    prompt = f""" {place}

Please output using the following format in English:
{{
    "description_reviews_rating_summary_in_brief_polite": "<summary>",
    "answer_to_extra_question_if_exists": "<answer>",
    "satisfication_score_to_extra_question": "<1-5>"
}}
"""
    del place['extra_questoin']
    #print(prompt)
    response = get_chatgpt_response([{'role': 'user', 'content': prompt}], model='gpt-3.5-turbo', temperature=0)['content']    
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
        details['summary'] = " ".join(list(summarized_output.values())[:2])
        #details['place_score'] = list(summarized_output.values())[-1]
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
    "Search locations and destinations, returning top_n results, calculate their distance matrix in km, and save to file.",
    '"search_keyword": "<Examples:강릉 여행지, 제주 가볼만한 곳, 신림 근처, 고기 맛집; avoid ranking numbers; instead use top_n arg>", '
    '"filename": "<yaml_filename>", "top_n": "<default:10>", "extra_request": "<Examples:뷰 좋은 곳, 주차 가능한가요?>"',
)
def search_places(search_keyword, filename, top_n=5, extra_request=""):
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
        return get_place_details(place_url, extra_request)
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

    with open(filename, 'w') as f:
        yaml.dump(results, f, allow_unicode=True)
    
    # place name 리스트 추출
    place_names = [f"{res['name']}({res['type']})" for res in results['candidates']]
    
    return f"The details of the found places:{place_names} have been written to {filename}. Information related to extra_request(if exists) has also been written."

if __name__ == '__main__':
    print(search_places('부산 여행지', 'busan_top5.yaml', top_n=5))
    
    #get_place_details('https://m.place.naver.com/place/11555552')
    #get_place_details('https://m.place.naver.com/place/1586994430')
    
    #get_place_details('https://m.place.naver.com/place/37942980')
    #get_place_details('https://m.place.naver.com/place/11658148')
    #get_place_details('https://m.place.naver.com/place/998885728')
    #get_place_details('https://m.place.naver.com/place/11859878')
    