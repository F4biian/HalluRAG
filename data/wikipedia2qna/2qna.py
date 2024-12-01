########################################################################################
# IMPORTS

from dotenv import load_dotenv
import json
import os
import re
import tiktoken
import traceback
import datetime
from tqdm import tqdm
from difflib import SequenceMatcher
from data.wikipedia.analyze_articles import get_useful_articles
from typing import Tuple
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.callbacks import get_openai_callback

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
QNA_FILE = os.path.join(CURR_DIR, "qna_per_passage_123.json")
MODEL = "gpt-4o-2024-05-13" # "gpt-3.5-turbo-0125"
SIMPLIFICATION_REGEX = r'\s|!|"|#|\$|%|&|\'|\(|\)|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|{|\||}|~'
LOG_FILE = os.path.join(CURR_DIR, "log.log")

# Loading env variables
load_dotenv()

# Get all those articles that contain only passages regarded as "useful"
articles = get_useful_articles()

# Setup LLM
llm = ChatOpenAI(model=MODEL, temperature=0.0, max_tokens=1000, model_kwargs={})

def log(msg: str) -> None:
    with open(LOG_FILE, "a") as file:
        file.write(f"[{datetime.datetime.utcnow()}] {msg}\n")

def similarity_str(str1: str, str2: str) -> float:
    """
    Returns a value between 0 and 1 indicating how similar the two string are.
    """
    return SequenceMatcher(None, str1, str2).ratio()

def extract_json_snippet(string: str) -> Tuple[str, dict]:
    # Define a regular expression pattern to match JSON snippets
    json_pattern = r'```json(.*?)```'

    # Use re.DOTALL to match across multiple lines
    match = re.search(json_pattern, string, re.DOTALL)

    if match:
        # Extract the JSON snippet
        json_str = match.group(1)

        try:
            # Load the JSON string into a dictionary
            return "Success", json.loads(json_str)
        except json.JSONDecodeError as e:
            return f"Error decoding JSON: {e}", {}
    else:
        try:
            # Load the JSON string into a dictionary
            return "Success", json.loads(string)
        except json.JSONDecodeError as e:
            return "No JSON snippet found in the text.", {}

def get_token_count_from_template(chat_prompt: ChatPromptTemplate) -> int:
    formatted = chat_prompt.format_messages(**{})
    input_message = formatted[0].content + "\n" + formatted[1].content # assuming two messages (system and human)
    encoding_gpt3_5_turbo = tiktoken.encoding_for_model(MODEL)
    input_tokens = len(encoding_gpt3_5_turbo.encode(input_message))
    return input_tokens

def get_qna_from_passage(art: dict, passage: dict) -> Tuple[str, str]:
    title = art["title"]

    # Extract the section's text before the given passage
    section_before_passage = art["content"][:passage["start"]].split("==\n")[-1].strip()

    # Extract passage from article
    passage_text = art["content"][passage["start"]:passage["end"]].strip()

    system_msg = "You are perfect at creating a question based on a sentence and its previous context. You also cite the answer to those questions from the given sentence."
    prompt = (
        "### PREVIOUS CONTEXT\n"
        f"Title: '{title}'\n"
        f"{section_before_passage}\n\n"
        "### SENTENCE\n"
        f"{passage_text}\n"
        "\n"
        "### OBJECTIVE\n"
        f"Write a question solely based on the given SENTENCE. This SENTENCE contains the definite answer which you also quote. This quote is definitely part of the SENTENCE. The question does not have the same wording as the SENTENCE. The question is 'globally' phrased and not 'locally', meaning that the question can be asked in a Retrieval Augmented Generation application.\n"
        "\n"
        "### RESPONSE\n"
        "The json format of your response should look like this:\n"
        "```json\n"
        "{{\n"
        "  \"answer_quote\": <answer copied from the sentence (as brief as possible)>,\n"
        "  \"question\": <question that is answer with the answer_quote>\n"
        "}}\n"
        "```\n"
        "Ensure your response can be parsed using Python json.loads\n"
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", prompt)
    ])

    # Build and invoke chain
    chain = chat_prompt | llm
    response = chain.invoke({})
    answer = response.content.strip()

    _, json_answer = extract_json_snippet(answer)

    return section_before_passage, passage_text, json_answer

qna_data = []

def save_data() -> None:
    with open(QNA_FILE, "w") as file:
        json.dump(qna_data, file, indent=4, ensure_ascii=False)

pbar = tqdm(total=len(articles))

no_answer_from = []

log("STARTING")

with get_openai_callback() as cb:
    for art_i, art in enumerate(articles):
        for passage_i, passage in enumerate(art["passage_data"]):
            try:
                # Ask GPT3.5-Turbo for a question and answer_quote
                section_before_passage, passage_text, json_answer = get_qna_from_passage(art, passage)
                # total_costs += cb.total_cost

                # If response contains wished data...
                if "answer_quote" in json_answer and "question" in json_answer:
                    answer_quote = json_answer["answer_quote"].strip()

                    answer_quote_simplified = re.sub(SIMPLIFICATION_REGEX, '', answer_quote.lower())
                    passage_text_simplified = re.sub(SIMPLIFICATION_REGEX, '', passage_text.lower())

                    # If answer_quote is really contained in sentence (and not made up or in context)...
                    if answer_quote_simplified in passage_text_simplified:
                        # Add this to data
                        qna_data.append({
                            "useful_art_i": art_i,
                            "useful_passage_i": passage_i,
                            "article_title": art["title"],
                            "passage_start": passage["start"],
                            "passage_end": passage["end"],
                            "context": section_before_passage,
                            "passage_text": passage_text,
                            "question": json_answer["question"],
                            "answer_quote": answer_quote,
                        })
                    else:
                        log(f"[{art_i}; {passage_i}] Did not find answer_quote '{answer_quote}'\nin passage_text '{passage_text}'!")
                        no_answer_from.append({
                            "useful_art_i": art_i,
                            "useful_passage_i": passage_i,
                            "reason": f"Did not find answer_quote '{answer_quote}'\nin passage_text '{passage_text}'!"
                        })
                else:
                    log(f"[{art_i}; {passage_i}]  No valid json!")
                    log(str(json_answer))
                    no_answer_from.append({
                        "useful_art_i": art_i,
                        "useful_passage_i": passage_i,
                        "reason": f"No valid json!"
                    })
            except KeyboardInterrupt:
                log(f"[{art_i}; {passage_i}] Saving...")
                save_data()
                log(f"[{art_i}; {passage_i}] Exiting...")
                exit()
            except:
                log(f"[{art_i}; {passage_i}] Exception:")
                log(f"[{art_i}; {passage_i}] {traceback.format_exc()}")
                no_answer_from.append({
                    "useful_art_i": art_i,
                    "useful_passage_i": passage_i,
                    "reason": f"Exception: {traceback.format_exc()}"
                })

        pbar.update()
        pbar.set_description(f"Costs: {cb.total_cost}")

pbar.refresh()
pbar.close()

log(f"No answers from: {no_answer_from}")

save_data()