"""Google search command for Autogpt."""
from __future__ import annotations

import json

from duckduckgo_search import ddg

from autogpt.commands.command import command
from autogpt.config import Config
#import os
import requests

import math
import numpy as np

CFG = Config()
from . import URL_MEMORY

@command("google", "Google Search", '"query": "<query>"', not CFG.google_api_key)
def google_search(query: str, num_results: int = 10) -> str:
    """Return the results of a Google search

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """
    global URL_MEMORY
    search_results = []
    if not query:
        return json.dumps(search_results)

    results = ddg(query, max_results=num_results)
    if not results:        
        return json.dumps(search_results)

    for j in results:
        url_alias = f'URL_{len(URL_MEMORY)}'
        URL_MEMORY[url_alias] = j['href']
        j['href'] = url_alias
        del j['body']
        search_results.append(j)

    results = json.dumps(search_results, ensure_ascii=False, indent=4)
    return safe_google_results(results)


@command(
    "google",
    "Google Search",
    '"query": "<query>"',
    bool(CFG.google_api_key) and bool(CFG.custom_search_engine_id),
    "Configure google_api_key and custom_search_engine_id.",
)
def google_official_search(query: str, num_results: int = 10) -> str | list[str]:
    """Return the results of a Google search using the official Google API

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """

    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    global URL_MEMORY
    try:
        # Get the Google API key and Custom Search Engine ID from the config file
        api_key = CFG.google_api_key
        custom_search_engine_id = CFG.custom_search_engine_id

        # Initialize the Custom Search API service
        service = build("customsearch", "v1", developerKey=api_key)

        # Send the search query and retrieve the results
        result = (
            service.cse()
            .list(q=query, cx=custom_search_engine_id, num=num_results)
            .execute()
        )
        search_results = []
        # Extract the search result items from the response
        results = result.get("items", [])

        # Create a list of only the URLs from the search results
        #search_results_links = [item["link"] for item in search_results]
        
        for res in results:
            url_alias = f'URL_{len(URL_MEMORY)}'
            URL_MEMORY[url_alias] = res['link']
            res['link'] = url_alias
            res = {k:v for k,v in res.items() if k in ['title', 'link']}
            search_results.append(res)

    except HttpError as e:
        # Handle errors in the API call
        error_details = json.loads(e.content.decode())

        # Check if the error is related to an invalid or missing API key
        if error_details.get("error", {}).get(
            "code"
        ) == 403 and "invalid API key" in error_details.get("error", {}).get(
            "message", ""
        ):
            return "Error: The provided Google API key is invalid or missing."
        else:
            return f"Error: {e}"
    # google_result can be a list or a string depending on the search results

    # Return the list of search result URLs
    results = json.dumps(search_results, ensure_ascii=False, indent=4)
    return safe_google_results(results)


def safe_google_results(results: str | list) -> str:
    """
        Return the results of a google search in a safe format.

    Args:
        results (str | list): The search results.

    Returns:
        str: The results of the search.
    """
    if isinstance(results, list):
        safe_message = json.dumps(
            [result.encode("utf-8", "ignore").decode("utf-8") for result in results]
        )
    else:
        safe_message = results.encode("utf-8", "ignore").decode("utf-8")
    return safe_message

@command(
    "get_textsearch_results_and_distances",
    "Google place textsearch and get distance_matrix (in km)",
    '"place_names": "<place_name_1>, <place_name_2>, ..."',
    bool(CFG.google_api_key),
)
def get_textsearch_results_and_distances(place_names: str, sort_by: str = "prominence") -> str:
    api_key = CFG.google_api_key
    place_names = [x.strip() for x in place_names.split(',')]  # 여러 장소를 리스트로 변환

    candidates = []
    for place_name in place_names:
        url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={place_name}&key={api_key}&rankby={sort_by}"
        result = requests.get(url)
        json_obj = result.json()

        if json_obj['results']:  # 검색 결과가 있는 경우
            candidate = json_obj['results'][0]  # 첫 번째 결과만 사용
            candidate = {key:val for key, val in candidate.items() if key in [
                'geometry', 'name', 'price_level', 'rating', 'types', 'user_ratins_total', 'formatted_address']}
            candidate['location'] = candidate['geometry']['location']
            #candidate['city_and_province'] = " ".join(candidate['formatted_address'].split())
            del candidate['geometry']#, candidate['formatted_address']
            candidates.append(candidate)

    num_places = len(candidates)
    distance_matrix = np.zeros((num_places, num_places))

    for i in range(num_places):
        for j in range(i+1, num_places):
            place1 = candidates[i]
            place2 = candidates[j]

            lat1, lon1 = place1["location"]["lat"], place1["location"]["lng"]
            lat2, lon2 = place2["location"]["lat"], place2["location"]["lng"]

            distance = calculate_distance(lat1, lon1, lat2, lon2)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance  # 대칭성을 이용
    
    # Remove 'location' field from each candidate after calculating the distance
    for candidate in candidates:
        del candidate['location']
    
    distance_matrix = distance_matrix.astype(int)
    distance_matrix_str = np.array2string(distance_matrix)

    return json.dumps({"candidates": candidates, "distance_matrix": distance_matrix_str}, ensure_ascii=False)

#@command(
#    "get_search_results_and_distances",
#    "Google place textsearch and get distance_matrix (in km)",
#    '"place_name": "<place_name>"',
#    bool(CFG.google_api_key),
#)
def get_search_results_and_distances(place_name: str, num_results: int = 10, sort_by: str = "prominence") -> str:
    api_key = CFG.google_api_key
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={place_name}&key={api_key}&rankby={sort_by}"
    result = requests.get(url)
    json_obj = result.json()
    
    candidates = []
    for candidate in json_obj['results'][:num_results]:
        candidate = {key:val for key, val in candidate.items() if key in [
            'geometry', 'name', 'price_level', 'rating', 'types', 'user_ratins_total', 'formatted_address']}
        candidate['location'] = candidate['geometry']['location']        
        candidate['city_and_province'] = " ".join(candidate['formatted_address'].split())
        del candidate['geometry'], candidate['formatted_address']
        candidates.append(candidate)
    
    num_places = len(candidates)
    distance_matrix = np.zeros((num_places, num_places))
    
    for i in range(num_places):
        for j in range(i+1, num_places):
            place1 = candidates[i]
            place2 = candidates[j]

            lat1, lon1 = place1["location"]["lat"], place1["location"]["lng"]
            lat2, lon2 = place2["location"]["lat"], place2["location"]["lng"]

            distance = calculate_distance(lat1, lon1, lat2, lon2)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance  # 대칭성을 이용
    distance_matrix = distance_matrix.astype(int)    
    distance_matrix_str = np.array2string(distance_matrix)

    return json.dumps({"candidates": candidates, "distance_matrix": distance_matrix_str}, ensure_ascii=False)

# Haversine 공식을 이용해 두 점 간의 거리를 계산하는 함수
def calculate_distance(lat1, lon1, lat2, lon2):
    radius = 6371  # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

#@command(
#    "search_place",
#    "Google Search place",
#    '"place_name": "<place_name>"',
#    bool(CFG.google_api_key),
#)
def google_search_place(place_name: str, num_results: int = 10) -> str:
    api_key = CFG.google_api_key
    url = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={place_name}&inputtype=textquery&fields=place_id,name&key={api_key}"
    result = requests.get(url)
    json_obj = result.json()

    candidates = []
    for candidate in json_obj['candidates'][:num_results]:
        place_id = candidate['place_id']  # Get the place ID of the first result
        place_details = get_place_details(place_id)
        place_details['address_components'] = [comp['short_name'] for comp in place_details['address_components'][1:-2]]
        place_details['location'] = place_details['geometry']['location']
        candidate.update(place_details)
        del candidate['place_id'], candidate['geometry']
        candidates.append(candidate)
    return json.dumps(candidates, ensure_ascii=False)

#@command(
#    "get_place_details",
#    "Get Place Details (location, ratings, address)",
#    '"place_name": "<place_name>"',
#    bool(CFG.google_api_key),
#)
def get_place_details(place_name: str) -> str:
    api_key = CFG.google_api_key

    # Get place_id
    find_place_url = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={place_name}&inputtype=textquery&fields=place_id,name&key={api_key}"
    find_place_result = requests.get(find_place_url)
    json_obj_find_place = find_place_result.json()

    # Assuming we always take the first place found
    place_id = json_obj_find_place['candidates'][0]['place_id']

    # Filter the details
    place_details = get_place_details_from_place_id(place_id)
    place_details['address_components'] = [comp['short_name'] for comp in place_details['address_components'][1:-2]]
    place_details['location'] = place_details['geometry']['location']
    del place_details['place_id'], place_details['geometry']

    return json.dumps(place_details, ensure_ascii=False)

def get_place_details_from_place_id(place_id):
    api_key = CFG.google_api_key
    fields = "name,geometry,price_level,rating,types,user_ratings_total,formatted_address"
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields={fields}&key={api_key}"
    result = requests.get(url)
    json_obj = result.json()
    return json_obj['result']

@command(
    "google_nearbysearch",
    "Google Search nearby places with distances (in km)",
    '"place_name": "<place_name>", "radius": "<default:1000>", "type": "<google_place_api_type_only>", "keyword": "<keyword>"',
    bool(CFG.google_api_key),
)
def google_nearbysearch(place_name: str, radius: int=1000, type: str = '', keyword: str = '',num_results: int = 10) -> str:
    api_key = CFG.google_api_key
    
    # Fetch the latitude and longitude of the place_name
    url_textsearch = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={place_name}&key={api_key}"
    result_textsearch = requests.get(url_textsearch)
    json_obj_textsearch = result_textsearch.json()
    location = json_obj_textsearch['results'][0]['geometry']['location']
    latitude, longitude = location['lat'], location['lng']
    
    url_nearbysearch = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={latitude},{longitude}&radius={radius}&type={type}&keyword={keyword}&key={api_key}"
    result_nearbysearch = requests.get(url_nearbysearch)
    json_obj_nearbysearch = result_nearbysearch.json()
    
    candidates = []
    for candidate in json_obj_nearbysearch['results'][:num_results]:
        place_id = candidate['place_id']  # Get the place ID of the first result
        candidate = get_place_details_from_place_id(place_id)        
        candidates.append(candidate)
    
    # Calculate distance vector
    num_places = len(candidates)
    distances = np.zeros(num_places)
    
    for i in range(num_places):
        place = candidates[i]
        lat, lon = place["geometry"]["location"]["lat"], place["geometry"]["location"]["lng"]
        distance = calculate_distance(latitude, longitude, lat, lon)
        distances[i] = distance
        
    # Remove 'location' field from each candidate after calculating the distance
    for candidate in candidates:
        del candidate['geometry']
        
    distances_str = np.array2string(distances, precision=1, floatmode='fixed')

    return json.dumps({"candidates": candidates, "distances": distances_str}, ensure_ascii=False)
