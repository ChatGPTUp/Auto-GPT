import os
import json
import concurrent.futures
from pathlib import Path
import time
from datetime import datetime

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import tiktoken
from trafilatura import fetch_url, extract
from duckduckgo_search import DDGS
from googleapiclient.discovery import build

from autogpt.commands.command import command
from autogpt.config import Config
from autogpt.llm.llm_utils import create_chat_completion

CFG = Config()
if CFG.workspace_path is None:
    CFG.workspace_path = Path.cwd()

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
    driver.set_page_load_timeout(10)
    return driver

def count_tokens(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def split_text(text, max_tokens=500, overlap=0):
    tokenizer = tiktoken.get_encoding("cl100k_base")
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

def get_doc_from_url(url):
    html = fetch_url(url)
    doc = extract(html, favor_recall=True, output_format="json")
    if doc is None:
        driver = get_selenium_driver()
        try:
            driver.get(url)
        except TimeoutException:
            return
        time.sleep(3)
        doc = extract(driver.page_source, favor_recall=True, output_format="json")
        if doc is None:
            return
        driver.quit()
    print(f'BROWSING: {url}')
    return json.loads(doc)

def get_relevant_ids(urls, titles, snippets, goal, n=5):
    prompt = ""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, (url, title, snippet) in enumerate(zip(urls, titles, snippets)):
        prompt += f"""
```
Id: {i}
Url: {url}
Title: {title}
Snippet: {snippet}
```"""
    prompt += f"""Current date and time is {now}. Given above search results, Choose up to {n} webpages that will most likely help achieve goal: {goal}. Respond in ids separated by commas.
ids:
"""
    response = create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=CFG.fast_llm_model,
        temperature=0.,
    )
    try:
        relevant_ids = [int(x.strip()) for x in response.split(',')]
    except:
        relevant_ids = list(range(n))
    return relevant_ids

def get_urls(query, type, goal, n_urls=10):
    if ('GOOGLE_API_KEY' in os.environ) and ('CUSTOM_SEARCH_ENGINE_ID' in os.environ) and (type == 'text'):
        service = build("customsearch", "v1", developerKey=os.getenv('GOOGLE_API_KEY'))
        result = (
            service.cse()
            .list(q=query, cx=os.getenv('CUSTOM_SEARCH_ENGINE_ID'), num=n_urls)
            .execute()
        )
        results = result.get("items", [])
        urls = [x['link'] for x in results]
        titles = [x['title'] for x in results]
        snippets = [x['snippet'] for x in results]
    else:
        ddgs = DDGS()
        ddgs_gen = getattr(ddgs, type)(
            query,
            region="wt-wt",
            safesearch="Off",
        )
        urls = []
        titles = []
        snippets = []
        for i, ddgs in enumerate(ddgs_gen):
            if i >= n_urls:
                break
            if 'url' in ddgs:
                url = ddgs['url']
            elif 'href' in ddgs:
                url = ddgs['href']
            titles.append(ddgs['title'])
            snippets.append(ddgs['body'])
            urls.append(url)
    if type == 'text':
        rel_ids = get_relevant_ids(urls, titles, snippets, goal)
        urls = [urls[i] for i in rel_ids]
    return urls

def summarize_chunk(chunk, goal):
    prompt = f"""```
    {chunk}
    ```
    Given above text, concisely extract key information that is potentially related to goal "{goal}"."""
    summary = create_chat_completion(
        messages=[{'role': 'user', 'content': prompt}],
        model=CFG.fast_llm_model,
        temperature=0.,
    )
    return summary

def summarize_doc(doc, goal, chunk_size=3000, chunk_overlap=10, max_chunks=5):
    chunks = split_text(doc, max_tokens=chunk_size, overlap=chunk_overlap)[:max_chunks]
    def summarize_chunk_(chunk):
        return summarize_chunk(chunk, goal)
    while len(chunks) > 1:
        with concurrent.futures.ThreadPoolExecutor(len(chunks)) as executor:
            summaries = executor.map(summarize_chunk_, chunks)
        summary = "\n".join(summaries)
        chunks = split_text(summary, max_tokens=chunk_size, overlap=chunk_overlap)
    summary = summarize_chunk_(chunks[0])        
    return summary

def generate_data(titles, summaries, goal):
    prompt = f""""""
    for i, (title, summary) in enumerate(zip(titles, summaries)):
        prompt += f"""
```
Id: {i+1}
Title: {title}
Summary: {summary}
```
"""
    prompt += f"""Concisely extract key information that is potentially related to goal "{goal}" from the above articles. \
Merge information from multiple articles if related. \
Respond in following JSON format:
[
{{
    "key_information": "<key information 1>",
    "reference_ids": [<id 1>, <id 2>, ...]
}}
...
]
"""
    data = create_chat_completion(
        messages=[{'role': 'user', 'content': prompt}],
        model=CFG.fast_llm_model,
        temperature=0.,
    )
    data = json.loads(data)
    return data

def data2report(data, urls):
    report = "\n".join([f"- {item['key_information']} {item['reference_ids']}" for item in data])
    used_reference_ids = set()
    for item in data:
        used_reference_ids = used_reference_ids.union(set(item['reference_ids']))
    report += "\nReferences\n" + "\n".join([f"{i+1}. {url}" for i, url in enumerate(urls) if (i+1) in used_reference_ids])
    return report

def feedback_info(info, goal):
    prompt = f"""```
{info}
```
Provide one sentence concise feedback for the above search result with respect to goal: "{goal}"."""
    feedback = create_chat_completion(
        messages=[{'role': 'user', 'content': prompt}],
        model=CFG.fast_llm_model,
        temperature=0.,
    )
    return feedback

def urls2report(urls, goal, filename):
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     doc_dicts = list(executor.map(get_doc_from_url, urls))
    doc_dicts = [get_doc_from_url(url) for url in urls]
    doc_dicts, urls = zip(*[(doc_dict, url) for doc_dict, url in zip(doc_dicts, urls) if doc_dict is not None])
    titles = [doc_dict['title'] for doc_dict in doc_dicts]
    docs = [doc_dict['text'] for doc_dict in doc_dicts]
    def summarize_doc_(doc):
        return summarize_doc(doc, goal)
    with concurrent.futures.ThreadPoolExecutor(len(docs)) as executor:
        summaries = list(executor.map(summarize_doc_, docs))
    data = generate_data(titles, summaries, goal)
    report = data2report(data, urls)
    with open(os.path.join(CFG.workspace_path, filename), "w") as f:
        f.write(report)
    feedback = feedback_info("\n".join([f"- {item['key_information']} {item['reference_ids']}" for item in data]), goal)
    return f"Wrote report at {filename}. \n {feedback}"

@command(
    "news_search",
    "Search news articles with keyword and save report",
    '"keyword": "<keyword>", "goal": "<goal to achieve via searching>", "filename": "<filename to save report>"',
)
def news_search(keyword, goal, filename):
    urls = get_urls(keyword, 'news', goal)
    return urls2report(urls, goal, filename)

@command(
    "google",
    "Search internet with keyword and save report",
    '"keyword": "<keyword>", "goal": "<goal to achieve via searching>", "filename": "<filename to save report>"',
)
def google(keyword, goal, filename):
    urls = get_urls(keyword, 'text', goal)
    return urls2report(urls, goal, filename)