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

def get_reviews2(soup, query):
    sections = soup.find_all('div', class_='place_section')
    review_section = None
    for section in sections:
        header = section.find(class_='place_section_header')
        if header and header.find(string=True, recursive=False) == '리뷰':
            review_section = section
            break
    if review_section:
        reviews = []
        li_elements = review_section.find('div', {'class': 'place_section_content'}).find('ul').find_all('li')
        for li in li_elements[:5]:
            review_text_element = li.find('a', class_='xHaT3').find('span')
            if review_text_element:  # 'span' 태그가 존재하면
                review = review_text_element.get_text(strip=True)
                reviews.append(review)
        if len(reviews) > 0:            
            reviews = summarize_reviews(reviews, query)
            return reviews

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

def get_place_details_org(place_url):
    print(f'BROWSING: {place_url}')
    driver = get_selenium_driver()
    details = {
        'name': None,
        'url': place_url,
        'type': None,
        'info': None,
        'rating': None,
        'n_visitor_reviews': None,
        'n_blog_reviews': None,
        'reviews': None,
    }
    
    driver.get(place_url)
    time.sleep(2)
    place_type = driver.current_url.split('/')[3]
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    # name / type
    spans = soup.find('div', {'id': '_title'}).find_all('span')
    details['name'] = spans[0].text
    if len(spans) > 1:
        details['type'] = spans[1].text
    # info
    info_section = soup.find_all('div', {'class': 'place_section_content'})[0]
    divs = info_section.find('div').find_all('div', recursive=False)
    info = [' '.join(div.find_all(string=True, recursive=True))[:100] for div in divs]
    #if do_summarize_info:
    #    info = summarize_info(info)
    details['info'] = info
    # ratings
    for span in soup.find('div', {'class': 'place_section'}).find_all('span'):
        text = span.text
        if text.startswith('별점') and (span.find('em') is not None):
            details['rating'] = text.replace('별점', '').strip()
        if text.startswith('방문자리뷰') and (span.find('em') is not None):
            details['n_visitor_reviews'] = text.replace('방문자리뷰', '').strip()
        elif text.startswith('블로그리뷰') and (span.find('em') is not None):
            details['n_blog_reviews'] = text.replace('블로그리뷰', '').strip()
    # reviews
    driver.get(f'{place_url}/review/visitor')
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    reviews = get_reviews(soup)
    details['reviews'] = reviews
    
    driver.get(f'{place_url}/photo')
    time.sleep(2)
    if 'photo' in driver.current_url:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        photo = soup.find('div', {'class': 'place_section_content'}).find('img')
        photo = shorten_url(photo['src'])
        details['photo'] = photo
    driver.quit()
    return details

def places2txt(places):
    texts = []
    for place in places:
        place_texts = []
        for k, v in place.items():
            if isinstance(v, list):
                v = '\n'.join(v)
                place_texts.append(f'{k}:\n{v}')
            else:
                place_texts.append(f'{k}: {v}')
        texts.append("\n".join(place_texts))
    return "\n\n".join(texts)


def get_reviews(soup, n_res=10):
    reviews_list = []
    
    reviews = soup.find_all('li', {'class': 'YeINN'})
    
    for review in reviews[:10]:
        review_dict = {}

        # 사용자 이름 추출
        #username = review.find('div', {'class': 'sBWyy'}).text
        #review_dict['username'] = username

        # 리뷰 텍스트 추출
        review_text = review.find('span', {'class': 'zPfVt'}).text
        review_text = " ".join(review_text.split())
        #review_dict['review_text'] = " ".join(review_text.split())
        
        # 이미지 URL 추출
        #images = review.find_all('div', {'class': 'K0PDV _img fKa0W'})
        #image_urls = [shorten_url(image['style'].split('"')[1]) for image in images]
        #review_dict['image_urls'] = image_urls

        # 추가로 필요한 정보를 추출하는 코드를 여기에 작성하실 수 있습니다.

        reviews_list.append(review_text)
    
    return reviews_list


def get_place_details(place_url):
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
    details['operating_hours'] = soup.select_one('.O8qbU.pSavy .A_cdD .U7pYf span.place_blind').text if soup.select_one('.O8qbU.pSavy .A_cdD .U7pYf span.place_blind') else None
        
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

    return details

@command(
    "search_places",
    "Search locations and destinations, returning top_n results, calculate their distance matrix in km, and save to file.",
    '"search_keyword": "<Examples:강릉 여행지, 제주 가볼만한 곳, 신림 근처; avoid ranking numbers; use top_n arg>", '
    '"filename": "<yaml_filename>", "top_n": "<default:10>"',
)
def search_places(search_keyword, filename, top_n=5):    
    driver = get_selenium_driver()
    naver_map_search_url = f'https://m.map.naver.com/search2/search.naver?query={search_keyword}'
    driver.get(naver_map_search_url)
    time.sleep(2)
    elements = driver.find_elements_by_css_selector('li._item._lazyImgContainer')[:top_n]
    if len(elements) == 0:
        driver.quit()
        return f"No results found for {search_keyword}. Try simpler place_query."
    def get_place_details_(place_url):
        return get_place_details(place_url)
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
    
    # place별 name, type만 넘기도록 함
    #{k:v for k, v in results.items() if k in ['name', 'type']}
    return_results = [{k:v for k, v in res.items() if k in ['name', 'type']} for res in results['candidates']]    
    return f"The details of the found places:{return_results} have been written to {filename}"

if __name__ == '__main__':
    print(search_places('속초 맛집', '속초맛집.yaml'))
    
    #get_place_details('https://m.place.naver.com/place/37942980')
    #get_place_details('https://m.place.naver.com/place/11658148')
    #get_place_details('https://m.place.naver.com/place/998885728')
    #get_place_details('https://m.place.naver.com/place/11859878')
    