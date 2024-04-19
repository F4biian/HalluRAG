# This should be called in the HalluRAG directory due to an import error that would occur otherwise.
# e.g.: python3 data/dataset/wikipedia_scraper.py

########################################################################################
# IMPORTS

import os
import pandas as pd
import json
from typing import List, Dict, Union, Any
from tqdm import tqdm
from mediawiki import MediaWiki
from mediawiki.exceptions import PageError, DisambiguationError
from pprint import pprint
import time
import requests
from bs4 import BeautifulSoup
import re
import urllib
import mwparserfromhell
import traceback
import datetime

from models.utils import sentence_split

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
ARTICLES_DIR = os.path.join(CURR_DIR, "articles")
LOG_FILE = os.path.join(CURR_DIR, "log.log")
REFERENCE_PATTERN = r'<ref(?:\s*name=\s*"?([^">\/]*)"?\s*)?>(.*?)<\/ref>|<ref(?:\s*name=\s*"?([^">]*)"?\s*)?\/>'
ADD_WHITESPACE_PATTERN = r'(?!\s*<)(\s*)'

wikipedia = MediaWiki(lang="en", rate_limit=True)

def log(msg: str) -> None:
    with open(LOG_FILE, "a") as file:
        file.write(f"[{datetime.datetime.utcnow()}] {msg}\n")

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

    got_response = False
    date = None
    attempts = 5

    while not got_response and attempts > 0:
        attempts -= 1
        response = requests.get(wiki_api_url, params=params)
        data = response.json()

        if response.ok:
            try:
                # Try to extract timestamp of very first revision
                pages = data["query"]["pages"]
                timestamp = pages[list(pages.keys())[0]]["revisions"][0]["timestamp"]

                # Convert to timezone unaware timestamp
                date = pd.to_datetime(timestamp).replace(tzinfo=None)
                got_response = True
            except:
                pass
        else:
            # Wait if request was not successful
            time.sleep(5)
    
        # Delay for rate limitting
        time.sleep(0.1)

    return date

def reference_to_json(ref_info: str) -> Dict[str, Any]:
    data = {}

    ref_info = ref_info.strip()

    if "{{" in ref_info and "}}" in ref_info:
        ref_info = ref_info.split("{{")[1].split("}}")[0]

        fields = ref_info.split("|")
        for field_index, field in enumerate(fields):
            # If the field indicates what type of citation/source it is
            if field_index == 0 and "=" not in field:
                data["ref_label"] = field.strip()
            # Otherwise...
            elif "=" in field:
                key, value = field.split('=', 1)
                data[key.strip()] = value.strip()
    
    return data

def extract_all_references(wikitext: str, url: str=None, verbose: bool=False) -> List[Dict[str, Any]]:
    # Find references in the wikitext string
    ref_matches = re.finditer(REFERENCE_PATTERN, wikitext, re.DOTALL)

    references_to_remove = set()

    reference_values_by_name = {}
    references = []

    # Extract all references
    for ref_match in ref_matches:
        starts_at, ends_at = ref_match.span()

        # First case: either <ref>ref_info_here</ref>, or <ref name="ref_name_here">ref_info_here</ref>
        ref_name_init = ref_match.group(1)
        ref_info = ref_match.group(2)

        # Second case: <ref name="ref_name_here">
        ref_name_reference = ref_match.group(3)

        # Index where the sentence/passage that the reference is associated with ends
        anchor_index = starts_at
        index_changed = True
        while index_changed:
            for ref in references:
                if ref["ends_at"] == anchor_index:
                    anchor_index = ref["starts_at"]
                    index_changed = True
                    break
            else:
                index_changed = False

        # If first case (the reference contains infomation)...
        if ref_info is not None:
            # ...and if the reference is associated with a name/id...
            if ref_name_init is not None:
                # ...store it in a dict so it can be accessed for further references with the same name/id.
                if ref_name_init not in reference_values_by_name:
                    ref_as_json = reference_to_json(ref_info)
                    reference_values_by_name[ref_name_init] = ref_as_json
                    references.append({
                        "name": ref_name_init,
                        "starts_at": starts_at,
                        "ends_at": ends_at,
                        "anchor_index": anchor_index,
                        "values": ref_as_json
                    })
                else:
                    if ref_info.strip():
                        if ref_name_init not in references_to_remove:
                            references_to_remove.add(ref_name_init)
                        if verbose:
                            log(f"Duplicate reference declaration for '{ref_name_init}' ({url})!")
                    else:
                        # Special case: ref_info is not None but empty due to false formatting (e.g. <ref name="name_here"></ref>)
                        references.append({
                            "name": ref_name_init,
                            "starts_at": starts_at,
                            "ends_at": ends_at,
                            "anchor_index": anchor_index,
                            "values": reference_values_by_name[ref_name_init]
                        })

            # If the reference is not linked to a previously declared reference...
            else:
                # ...already store it in the overall reference list.
                references.append({
                    "name": None,
                    "starts_at": starts_at,
                    "ends_at": ends_at,
                    "anchor_index": anchor_index,
                    "values": reference_to_json(ref_info)
                })

        # ...or if second case (the reference is linked to a previously declared reference)...
        elif ref_name_reference is not None:
            # ...look up the linked reference in the dict and add the reference to the reference list.
            if ref_name_reference in reference_values_by_name:
                references.append({
                    "name": ref_name_reference,
                    "starts_at": starts_at,
                    "ends_at": ends_at,
                    "anchor_index": anchor_index,
                    "values": reference_values_by_name[ref_name_reference]
                })
            else:
                references.append({
                    "name": ref_name_reference,
                    "starts_at": starts_at,
                    "ends_at": ends_at,
                    "anchor_index": anchor_index,
                    "values": None
                })

        # ...otherwise...
        else:
            if verbose:
                log(f"\nThis reference format seems not to be covered by the code:")
                for i, group in enumerate(ref_match.groups()):
                    log(f"Group Index {i+1}: {group}")

    # If references were used before they have been declared, fill those gaps:
    for i in range(len(references)):
        ref = references[i]
        if ref["values"] is None:
            if ref["name"] in reference_values_by_name:
                # If the reference exists, insert it into the reference dict...
                references[i]["values"] = reference_values_by_name[ref["name"]]
            else:
                # ...otherwise insert an empty dict.
                references[i]["values"] = {}

    # Delete duplicate references (safety first)
    for ref in references.copy():
        if ref["name"] in references_to_remove:
            del ref

    return len(references_to_remove) > 0, references

def get_passage_start(article: str, candidate: str) -> int:
    # Look for the same beginnings in the article...
    for i in range(1, len(candidate)):
        match_i = 0
        last_match = None

        # Go through every match...
        for match in re.finditer(re.escape(candidate[:i]), article):
            # If there is more than one match, cancel loop and continue with a longer beginning...
            if match_i >= 1:
                break
            match_i += 1
            last_match = match

        # If there is only one match (beginning is unique)...
        else:
            # ...return start index of the candidate.
            if last_match is not None:
                return last_match.start()
            else:
                return None
        
    # Special case: Passage's start is not unique, thus cannot be found
    return None

def get_passage_end(article: str, candidate: str) -> int:
    # Look for the same ending in the article...
    for i in range(1, len(candidate)):
        match_i = 0
        last_match = None

        # Go through every match...
        for match in re.finditer(re.escape(candidate[-i:]), article):
            # If there is more than one match, cancel loop and continue with a longer ending...
            if match_i >= 1:
                break
            match_i += 1
            last_match = match

        # If there is only one match (ending is unique)...
        else:
            # ...return end index of the candidate.
            if last_match is not None:
                return last_match.end()
            else:
                return None
        
    # Special case: Passage's end is not unique, thus cannot be found
    return None

def get_passage_span(article: str, candidates: List[str], anchor_index: int, increasing_index: int=1) -> int:
    candidates_that_still_fit = []

    # Compare increased window at the end of both string and check if they match
    for candidate in candidates:
        if article[anchor_index-increasing_index:anchor_index] in candidate:
            candidates_that_still_fit.append(candidate)

    # Repeat process, until the last candidate is unique
    if len(candidates_that_still_fit) == 1:
        starts_at = get_passage_start(article, candidates_that_still_fit[0])
        if starts_at:
            # return (starts_at, starts_at + len(candidates_that_still_fit[0]))
            if starts_at + len(candidates_that_still_fit[0]) == anchor_index:
                return starts_at, anchor_index
            else:
                return starts_at, get_passage_end(article, candidates_that_still_fit[0])
        else:
            return None, None
    elif len(candidates_that_still_fit) == 0:
        # Special case: Passage's end is not unique, thus cannot be found
        return None, None
    else:
        return get_passage_span(article, candidates_that_still_fit, anchor_index, increasing_index+1)

def remove_references(text: str) -> str:
    return re.sub(REFERENCE_PATTERN, '', text, flags=re.DOTALL)

def parse_wikitext(wikitext) -> str:
    parsed_text = mwparserfromhell.parse(wikitext)
    return parsed_text.strip_code()

def add_spaces_after_references(wikitext: str) -> str:
    def add_space(match):
        reference = match.group(0)
        whitespace_after_reference = match.group(5)
        if not whitespace_after_reference:
            return reference + " "
        else:
            return reference
        
    return re.sub(f'({REFERENCE_PATTERN}){ADD_WHITESPACE_PATTERN}', add_space, wikitext)

def get_article_data_from(url: str, debug_mode: bool=True) -> Dict[str, Any]:
    # Extract the title from the URL
    parsed_url = urllib.parse.urlparse(url)
    title = urllib.parse.unquote(parsed_url.path.split('/')[-1])

    try:
        p = wikipedia.page(title, auto_suggest=False)
    except (PageError, DisambiguationError):
        # Page not found or not uniquely identifiable
        return None

    # Adding a whitespace behind a reference (if it does not have a trailing whitespace or "<" bracket already)
    # This is needed for a better sentence tokenization.
    wikitext = add_spaces_after_references(p.wikitext)

    if debug_mode:
        with open("content.txt", "w") as file:
            file.write(p.content)
        with open("wikitext.txt", "w") as file:
            file.write(wikitext)

    earliest_creation_date = None

    for lang_code in p.langlinks:
        lang_title = p.langlinks[lang_code]
        lang_wikipedia = MediaWiki(lang=lang_code, rate_limit=True)

        # Get creation date of article in this language
        creation_date = get_creation_date(lang_title, lang_wikipedia.api_url)

        # Update min creation date
        if creation_date is not None:
            if earliest_creation_date is None or earliest_creation_date > creation_date:
                earliest_creation_date = creation_date

    removed_duplicates, references = extract_all_references(wikitext, url, verbose=debug_mode)

    if debug_mode:
        if removed_duplicates:
            log(f"Removed duplicates for {url}!")

    wikitext_without_references = remove_references(wikitext)

    if debug_mode:
        if "<ref>" in wikitext_without_references or "ref>" in wikitext_without_references or "<ref " in wikitext_without_references or "ref >" in wikitext_without_references:
            log(f"<ref still in article of '{url}'!")
            log(f"len(references) = {len(references)}")
            with open(f"wikitext_without_references_{title}.txt", "w") as file:
                file.write(wikitext_without_references)
            with open(f"wikitext_with_references_{title}.txt", "w") as file:
                file.write(wikitext)
            exit()

    # Split article into sentences/lines
    passages_from_wikitext = []
    for line in wikitext_without_references.split("\n"):
        for sentence in sentence_split(line):
            # Add passage to list
            passages_from_wikitext.append(sentence)

    # Group references because a sentence might be associated with e.g. 3 references ("This could be a sentence like this.[1][2][7]"")
    #                                                           This index right here can be the value of (anchor_index) ^
    #                                                           The anchor_index can also be right inside a passage.
    references_grouped_by_passage = {}
    for ref in references:
        # Get start and end index of passage/sentence the reference is linked to
        passage_starts_at, passage_ends_at = get_passage_span(wikitext, passages_from_wikitext, ref["anchor_index"])

        # If both beginning and end were found...
        if passage_starts_at and passage_ends_at:
            # ...assign the reference to the corresponding passage.
            passage_id = (passage_starts_at, passage_ends_at)
            if passage_id not in references_grouped_by_passage:
                references_grouped_by_passage[passage_id] = []
            references_grouped_by_passage[passage_id].append(ref)

    passage_data = []

    for passage_start, passage_end in references_grouped_by_passage:
        passage_references = references_grouped_by_passage[(passage_start, passage_end)]
        passage_without_references = remove_references(wikitext[passage_start:passage_end])

        # Check if the passage contains a Wikipedia article
        contains_wikipedia_article = ("[[" in passage_without_references) and ("]]" in passage_without_references)

        earliest_access_date = None
        earliest_date = None
        earliest_archive_date = None
        ref_infos = []

        # Go through every references this passage is linked to
        for ref in passage_references:
            access_date = None
            date = None
            archive_date = None

            # Go through every field the references has
            for key in ref["values"]:
                value = ref["values"][key]
                modified_key = key.lower().strip().replace("-", "")

                # If the field indicates an access date...
                if modified_key == "accessdate":
                    try:
                        access_date = pd.to_datetime(value)
                        if earliest_access_date is None or earliest_access_date > access_date:
                            earliest_access_date = access_date
                    except:
                        if debug_mode:
                            log(f"Warning: Could not convert '{value}' to a datetime object for {url}!\nRef: {ref}")
                # or a date...
                elif modified_key == "date":
                    try:
                        date = pd.to_datetime(value)
                        if earliest_date is None or earliest_date > date:
                            earliest_date = date
                    except:
                        if debug_mode:
                            log(f"Warning: Could not convert '{value}' to a datetime object for {url}!\nRef: {ref}")
                # or an archive date...
                elif modified_key == "archivedate":
                    try:
                        archive_date = pd.to_datetime(value)
                        if earliest_archive_date is None or earliest_archive_date > archive_date:
                            earliest_archive_date = archive_date
                    except:
                        if debug_mode:
                            log(f"Warning: Could not convert '{value}' to a datetime object for {url}!\nRef: {ref}")
                elif "date" in modified_key:
                    log(f"Warning! Found another type of date: {key} for article {url}!\n Ref: {ref}")

            ref_infos.append({
                "key_count": len(ref["values"]),
                "ref_label": ref["values"]["ref_label"] if "ref_label" in ref["values"] else None,
                "access_date": str(access_date) if access_date else None,
                "date": str(date) if date else None,
                "archive_date": str(archive_date) if archive_date else None,
            })

        # "Render" wikitext to entirely human readable text as if it was shown on Wikipedia
        parsed_passage = parse_wikitext(passage_without_references).strip()

        # Find start and end index in real/parsed article (which will be used for RAG)
        start = get_passage_start(p.content, parsed_passage)
        end   =   get_passage_end(p.content, parsed_passage)

        if start is not None and end is not None:
            passage_data.append({
                "start": start,
                "end": end,
                "contains_article": contains_wikipedia_article,
                "earliest_access_date": str(earliest_access_date) if earliest_access_date else None,
                "earliest_archive_date": str(earliest_archive_date) if earliest_archive_date else None,
                "earliest_date": str(earliest_date) if earliest_date else None,
                "references": ref_infos
            })

    return {
        "url": url,
        "title": p.title,
        "revision_id": p.revision_id,
        "earliest_creation_date": str(earliest_creation_date) if earliest_creation_date else None,
        "content": p.content,
        "passage_data": passage_data,
        "removed_duplicates": removed_duplicates
    }

def get_newest_wikipedia_articles(end: str, start: str=None) -> List[Dict[str, str]]:
    # Convert strings to datetime objects
    end_date = pd.to_datetime(end)
    start_date = pd.to_datetime(start) if start else None

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

                    if start_date:
                        # If the creation date is younger than the given start date...
                        if created > start_date:
                            # ...ignore this article.
                            continue 

                    # If the article has been created before the given end date...
                    if created <= end_date:
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
                log("Response: {response}")
                log(f"New attempt ({attempts_left} left)...")
                time.sleep(60)

    return all_articles


if __name__ == "__main__":
    articles = get_newest_wikipedia_articles(start="2024-04-19", end="2024-04-15")

    log(f"Found {len(articles)} articles!")

    all_articles_with_data = []
    last_processed_date = pd.to_datetime(articles[0]["created"]).date()

    def save_articles():
        log("Saving...")
        with open(os.path.join(ARTICLES_DIR, f"articles_{str(last_processed_date)}.json"), "w") as file:
            json.dump(all_articles_with_data, file, indent=4)

    for art in tqdm(articles, desc="Retrieving Data"):
        try:
            # Save current list once a new date is getting processed
            processed_date = pd.to_datetime(art["created"]).date()
            if processed_date != last_processed_date:
                save_articles()
                all_articles_with_data.clear()
            last_processed_date = processed_date

            # Get passages and other data from the article
            article_data = get_article_data_from(art["href"])

            if article_data:
                all_articles_with_data.append({
                    "created_en": str(art["created"]),
                    **article_data
                })
        except KeyboardInterrupt:
            print("Exiting...")
            log("Exiting...")
            save_articles()
            exit()
        except:
            log(traceback.format_exc())
            log(f"Error occurred while processing {art['href']}")
    else:
        save_articles()
