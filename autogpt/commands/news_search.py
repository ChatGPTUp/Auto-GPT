import os
import json
import concurrent.futures

import tiktoken
from trafilatura import fetch_url, extract
from duckduckgo_search import DDGS

from autogpt.commands.command import command
from autogpt.config import Config
from autogpt.llm.llm_utils import create_chat_completion

CFG = Config()

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
    downloaded = fetch_url(url)
    doc = extract(downloaded, favor_recall=True, output_format="json")
    if doc is None:
        return None
    return json.loads(doc)

def get_urls(query, n_urls=10):
    ddgs = DDGS()
    ddgs_news_gen = ddgs.news(
        query,
        region="wt-wt",
        safesearch="Off",
    )
    urls = []
    for i, ddgs_news in enumerate(ddgs_news_gen):
        if i >= n_urls:
            break
        urls.append(ddgs_news['url'])
    # urls = get_relevant_urls(search_results)
    return urls

def get_relevant_urls(search_results):
    prompt = ""
    for i, item in enumerate(search_results):
        prompt += f"""
```
Id: {i}
Title: {item['title']}
```"""
    prompt += f"""Given above search results, group search results based on the specific event. Respond in JSON format:
[
{{"specific_event": "<event>", "ids": [<id1>, <id2>, ...]}},
...
]
"""
    response = create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=CFG.fast_llm_model,
        temperature=0.,
    )
    relevent_ids = [x['ids'][0] for x in json.loads(response)]
    relevent_urls = [search_results[i]['url'] for i in relevent_ids]
    return relevent_urls

def summarize_chunk(chunk, goal):
    prompt = f"""```
    {chunk}
    ```
    Summarize above content concisely, focusing on information potentially related to goal "{goal}".
    Summary:"""
    summary = create_chat_completion(
        messages=[{'role': 'user', 'content': prompt}],
        model=CFG.fast_llm_model,
        temperature=0.,
    )
    return summary

def summarize_doc(doc, goal, chunk_size=3000, chunk_overlap=10):
    chunks = split_text(doc, max_tokens=chunk_size, overlap=chunk_overlap)
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

@command(
    "news_search",
    "Search news articles with keyword and save report",
    '"keyword": "<keyword>", "goal": "<goal to achieve via searching>", "filename": "<filename to save report>"',
)
def news_search(keyword, goal, filename):
    urls = get_urls(keyword)
    doc_dicts = [get_doc_from_url(url) for url in urls]
    doc_dicts = [doc_dict for doc_dict in doc_dicts if doc_dict is not None]
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
    return f"Wrote report at {filename}"