########################################################################################
# IMPORTS

import os
import pandas as pd
import json
from typing import List, Dict, Union
from tqdm import tqdm
from mediawiki import MediaWiki
from pprint import pprint
import time
import requests
from bs4 import BeautifulSoup

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

wikipedia = MediaWiki(lang="en")

def get_creation_date(article_title: str, wiki_api_url: str) -> Union[None, "Timestamp"]:
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "timestamp",
        "rvlimit": 1,
        "rvdir": "newer",
        "titles": article_title,
        "format": "json"
    }

    response = requests.get(wiki_api_url, params=params)
    data = response.json()

    if response.ok:
        try:
            # Try to extract timestamp of very first revision
            pages = data["query"]["pages"]
            timestamp = pages[list(pages.keys())[0]]["revisions"][0]["timestamp"]

            # Convert to timezone unaware timestamp
            return pd.to_datetime(timestamp).replace(tzinfo=None)
        except:
            pass
    
    return None

def check_and_get_contents(title):
    p = wikipedia.page(title)

    earliest_creation_date = None

    # TODO: check other versions in other languages
    for lang_code in p.langlinks:
        lang_title = p.langlinks[lang_code]
        lang_wikipedia = MediaWiki(lang=lang_code)

        # Get creation date of article in this language
        creation_date = get_creation_date(lang_title, lang_wikipedia.api_url)

        # Update min creation date
        if earliest_creation_date is None or earliest_creation_date > creation_date:
            earliest_creation_date = creation_date

    # TODO: Hier weiter machen und den satz + Section extrahieren (bzw. x chars vorher und y chars nachher?!?)
    pprint(p.wikitext)

def get_newest_wikipedia_articles(since: str) -> List[Dict[str, str]]:
    # Convert string to datetime object
    since_date = pd.to_datetime(since)

    end_reached = False
    next_page = "https://en.wikipedia.org/w/index.php?title=Special:NewPages&offset=&limit=500"

    all_articles = []

    pbar = tqdm()
    pbar.set_description(f'Last created: {None} | Length: {len(all_articles)}')

    # As long as the end date has not been reached, scrape...
    while not end_reached:
        # Provide 5 attempts for reaching wikipedia
        attempts_left = 5
        while attempts_left > 0:
            attempts_left -= 1

            # Get wikipedia page of new articles
            response = requests.get(next_page)
            if response.ok:
                soup = BeautifulSoup(response.text, "html.parser")
                article_elements = soup.find_all('li', {'data-mw-revid': True})

                for art_el in article_elements:
                    # Extract date of creation
                    created = pd.to_datetime(art_el.find("span", {"class": "mw-newpages-time"}).text)

                    # If the article has been created before the given date...
                    if created <= since_date:
                        # ...stop collecting data
                        end_reached = True
                        break

                    # If the article has been created after the given date, add it to the list:
                    title_element = art_el.find("a", {"class": "mw-newpages-pagename"})
                    all_articles.append({
                        "created": str(created),
                        "title": title_element["title"],
                        "href": "https://en.wikipedia.org" + title_element["href"],
                    })

                # Extract the URL of the next page
                next_page = "https://en.wikipedia.org" + soup.find("a", {"class": "mw-nextlink"})["href"]
                pbar.update()
                pbar.set_description(f'Last created: {created} | Length: {len(all_articles)}')

                # Delay for reducing traffic per time
                time.sleep(0.3)
                break
            else:
                print("Response:", response)
                print(f"New attempt ({attempts_left} left)...")
                time.sleep(60)

    return all_articles

# articles = get_newest_wikipedia_articles(since="2024-02-22")
# print(articles)

t = check_and_get_contents('Tiden (newspaper, Arendal)')
# t = check_and_get_contents('Peace efforts during the Iran–Iraq War')
