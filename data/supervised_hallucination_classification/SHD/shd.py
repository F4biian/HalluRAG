########################################################################################
# IMPORTS

from dotenv import load_dotenv
import json
import os
import re
import traceback
from typing import Tuple
from pprint import pprint
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

from models.utils import sentence_split

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL = "gpt-3.5-turbo-0125"

# Loading env variables
load_dotenv()

# Setup GPT3.5-Turbo
llm = ChatOpenAI(model=MODEL, temperature=0.0, max_tokens=1000, model_kwargs={})

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

def classify_unsupervised(passage_texts: str, question: str, llm_output: str):
    system_msg = "You are given a QUESTION which is solely based on the CHUNKS. You are perfect at comparing the OUTPUT with the CHUNKS. One of the CHUNKS contains the correct answer to the QUESTION. The OUTPUT is the answer that you check for mistakes."
    prompt = (
        "### CHUNKS\n"
        f"{passage_texts}\n"
        "\n"
        "### QUESTION\n"
        f"{question}\n"
        "\n"
        "### OUTPUT\n"
        f"{llm_output}\n"
        "\n"
        "### OBJECTIVE\n"
        f"Compare the OUTPUT with the CHUNKS on word by word and by looking for mistakes or facts that are not part of the CHUNKS. You also classify the type of mistake in the OUTPUT:\n"
        "- NOT_GROUNDED: part of the OUTPUT is not grounded in the CHUNKS\n"
        "- CONFLICTING: part of the OUTPUT contradicts information from the CHUNKS\n"
        # "- NO_HELP: part of the OUTPUT does not relate to the QUESTION or OUTPUT does not refer to the CHUNKS\n"
        "After sharing your examination thoughts, for each mistake you always quote as briefly as possible the \"wrong\" part from the OUTPUT and the \"correct\" part from the CHUNKS. If there are no mistakes, the \"mistakes\" list is empty.\n"
        "\n"
        "### RESPONSE\n"
        "The json format of your response should look like this:\n"
        "```json\n"
        "{{\n"
        "  \"thoughts\": <briefly compare and examine if mistakes exist>,\n"
        "  \"has_mistakes\": <true if you found a mistake (either NOT_GROUNDED or CONFLICTING), false otherwise>,\n"
        "  \"mistakes\": [\n"
        "    {{\n"
        "      \"type\": <either NOT_GROUNDED or CONFLICTING>,\n"
        "      \"output_quote\": <wrong words quoted from the OUTPUT>,\n"
        "      \"chunks_quote\": <correct words quoted from the CHUNKS> # ignore \"chunks_quote\" if type is NOT_GROUNDED\n"
        "    }},\n"
        "    {{\n"
        "      \"type\": <either NOT_GROUNDED or CONFLICTING>,\n"
        "      \"output_quote\": <wrong words quoted from the OUTPUT>,\n"
        "      \"chunks_quote\": <correct words quoted from the CHUNKS> # ignore \"chunks_quote\" if type is NOT_GROUNDED\n"
        "    }},\n"
        "    ...\n"
        "  ]\n"
        "}}\n"
        "```\n"
        "Ensure your response can be parsed using Python json.loads\n"
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", prompt)
    ])

    print("#"*50)
    print(prompt)

    # Build and invoke chain
    chain = chat_prompt | llm
    response = chain.invoke({})
    answer = response.content.strip()

    print("#"*50)
    print(answer)
    print("#"*50)

    _, json_answer = extract_json_snippet(answer)

    pprint(json_answer)
    print("#"*50)

    return json_answer

def classify(title: str, chunk: str, chunk_index: int, question: str, answer_quote: str, llm_output_split: list, titles: list, answerable: bool):
    titles_str = "\n".join(f"Knowledge Chunk {title_i+1}: '{title}'" for title_i, title in enumerate(titles))

    section_str = ""
    sections = []
    section_count = 0
    for sent_i, sent in enumerate(llm_output_split):
        sections.append(sent)
        section_str += f"#### SECTION {sent_i+1}:\n{sent}\n"
        section_count += 1

    if answerable:
        conflicting_desc = "true, if the current SECTION contains information explicitly conflicting information from the NECESSARY CHUNK; otherwise false"
        # unverifiable_desc = "true, if the current SECTION contains information not mentioned in the NECESSARY CHUNK; otherwise false"
        no_clear_answer_desc = "true, if in the current SECTION the AI stated that it cannot access the information/chunk/knowledge, cannot find the information/chunk/knowledge, does not know the answer, explains what KNOWLEDGE CHUNKS were provided except for the NECESSARY CHUNK, or the AI stated anything similar; otherwise false"
    else:
        conflicting_desc = "true, if the current SECTION contains information explicitly conflicting information from the NECESSARY CHUNK; otherwise false"
        # unverifiable_desc = "true, if the current SECTION contains information not mentioned in the NECESSARY CHUNK; otherwise false"
        no_clear_answer_desc = "true, if in the current SECTION the AI stated that it cannot access the information/chunk/knowledge, cannot find the information/chunk/knowledge, does not know the answer, explains what KNOWLEDGE CHUNKS or topics were provided except for the NECESSARY CHUNK, or the AI stated anything similar; otherwise false"

    more_sections_format_text = ""

    if section_count == 1:
        more_sections_format_text += "\n"
    elif section_count == 2:
        more_sections_format_text += (
        ",\n"
        "  \"SECTION 2\": {{\n"
        "    \"section_content\": <repeat just SECTION 2 here>,\n"
        "    ...\n"
        "  }}\n"
        )
    elif section_count > 2:
        more_sections_format_text += (
        ",\n"
        "  ...\n"
        "  \"SECTION " + str(section_count) + "\": {{\n"
        "    \"section_content\": <repeat just SECTION " + str(section_count) + " here>,\n"
        "    ...\n"
        "  }}\n"
        )

    system_msg = f"You are perfect at assessing an AI RESPONSE to a QUESTION. The AI was given {len(titles)} KNOWLEDGE CHUNKS in order to give a response to the QUESTION. The QUESTION can only be answered using the NECESSARY CHUNK. {'This NECESSARY CHUNK was provided to the AI, so the AI should have been able to give the correct answer.' if answerable else 'This NECESSARY CHUNK was not provided to the AI, so the AI should not have been able to give the correct answer.'}"
    prompt = (
        f"### All KNOWLEDGE CHUNKS provided to AI\n"
        f"{titles_str}\n\n"
        "### QUESTION\n"
        f"{question}\n"
        "\n"
        f"### CONTENT of the NECESSARY CHUNK containing the real answer\n"
        f"Title: '{title}'\n"
        f"{chunk}\n\n"
        "### GROUND TRUTH ANSWER to the QUESTION\n"
        f"{answer_quote}\n"
        "\n"
        "### Was AI provided with the CONTENT of the NECESSARY CHUNK containing the real answer?\n"
        f"{'Yes, the AI was provided with the NECESSARY CHUNK needed for answering the QUESTION. Thus, the AI was able to give a correct answer.' if answerable else 'No, the AI was not provided with the NECESSARY CHUNK needed for answering the QUESTION. Thus, the AI was not able to give the correct answer and should hopefully responded with `I do not know` or something similar.'}\n"
        "\n"
        "### AI RESPONSE to the QUESTION \n"
        "The AI RESPONSE is divided into the following SECTIONs:\n"
        f"{section_str}\n"
        "\n"
        "### OBJECTIVE\n"
       f"{'Go through every SECTION and determine these criteria for each of them:' if section_count > 1 else 'For SECTION 1, determine these criteria:'}\n" 
       f"- conflicting: {conflicting_desc}\n"
    #    f"- unverifiable: {unverifiable_desc}\n"
       f"- no_clear_answer: {no_clear_answer_desc}\n"
        "Before giving the final boolean result, you first write down your very brief thoughts of the current criteria on the current SECTION and you always provide evidence of your thoughts by quoting the specific words.\n"
        "You solely refer to the section_content when evaluating a SECTION. Do not refer to other SECTIONs or section_contents!\n"
        "\n"
        "### YOUR RESPONSE\n"
        "The json format of your response should look like this:\n"
        "```json\n"
        "{{\n"
        "  \"SECTION 1\": {{\n"
       f"    \"section_content\": <repeat just SECTION 1 here>,\n"
        "    \"conflicting\": {{\n"
        "      \"thoughts\": <your extremely brief thoughts>,\n"
        "      \"necessary_chunk_quote\": <brief: conflicting words quoted from the NECESSARY CHUNK; keep this empty if section_content does not conflict with the NECESSARY CHUNK>,\n"
        "      \"section_quote\": <brief: conflicting words quoted from the current section_content; keep this empty if section_content does not conflict with the NECESSARY CHUNK>,\n"
        "      \"reflexion\": <your extremely brief thoughts on your thoughts and quotes: Are you thoughts still true? What do you set as the resulting boolen? You may change your previous evaluation and thoughts>,\n"
       f"      \"result\": <{conflicting_desc}>\n"
        "    }},\n"
    #     "    \"unverifiable\": {{\n"
    #     "      \"thoughts\": <your extremely brief thoughts>,\n"
    #     "      \"section_quote\": <brief: verifiable or unverifiable words quoted from the current section_content>,\n"
    #     "      \"reflexion\": <your extremely brief thoughts on your thoughts and quotes: Are you thoughts still true? What do you set as the resulting boolen? You may change your previous evaluation and thoughts>,\n"
    #    f"      \"result\": <{unverifiable_desc}>\n"
    #     "    }},\n"
        "    \"no_clear_answer\": {{\n"
        "      \"thoughts\": <your extremely brief thoughts>,\n"
        "      \"section_quote\": <brief: words quoted from the current section_content; keep this empty if the section_content is a clear answer>,\n"
        "      \"reflexion\": <your extremely brief thoughts on your thoughts and quotes: Are you thoughts still true? What do you set as the resulting boolen? You may change your previous evaluation and thoughts>,\n"
       f"      \"result\": <{no_clear_answer_desc}>\n"
        "    }}\n"
        "  }}"
        f"{more_sections_format_text}"
        "}}\n"
        "```\n"
        "Ensure your response can be parsed using Python json.loads\n"
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", prompt)
    ])

    print("#"*50)
    print(prompt)

    # Build and invoke chain
    chain = chat_prompt | llm
    response = chain.invoke({})
    answer = response.content.strip()

    print("#"*50)
    print(answer)
    print("#"*50)

    _, json_answer = extract_json_snippet(answer)

    pprint(json_answer)
    print("#"*50)

    return json_answer