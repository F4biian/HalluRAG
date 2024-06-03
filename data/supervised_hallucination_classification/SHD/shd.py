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

def classify(title: str, section_before_passage: str, passage_text: str, question: str, answer_quote: str, llm_output: str):
    system_msg = "You are given a QUESTION which is solely based on the SENTENCE. You are perfect at comparing the OUTPUT with the REFERENCE ANSWER. The REFERENCE ANSWER is the correct answer to the QUESTION. The OUTPUT is the answer that you check for mistakes."
    prompt = (
        "### PREVIOUS CONTEXT\n"
        f"Title: '{title}'\n"
        f"{section_before_passage}\n\n"
        "### SENTENCE\n"
        f"{passage_text}\n"
        "\n"
        "### QUESTION\n"
        f"{question}\n"
        "\n"
        "### REFERENCE ANSWER\n"
        f"{answer_quote}\n"
        "\n"
        "### OUTPUT\n"
        f"{llm_output}\n"
        "\n"
        "### OBJECTIVE\n"
        f"Compare the REFERENCE ANSWER with the OUTPUT on sentence-level by looking for mistakes or facts that are not part of the REFERENCE ANSWER. You also classify the type of mistake:\n"
        "- INCORRECT: OUTPUT is not entirely correct, contains mistake\n"
        "- NO_HELP: OUTPUT does not relate to the QUESTION or OUTPUT does not refer to the SENTENCE\n"
        "After sharing your examination thoughts, for each mistake you always quote as briefly as possible the \"wrong\" part from the OUTPUT and the \"correct\" part from the REFERENCE ANSWER. If there are no mistakes, the \"mistakes\" list is empty.\n"
        "\n"
        "### RESPONSE\n"
        "The json format of your response should look like this:\n"
        "```json\n"
        "{{\n"
        "  \"thoughts\": <briefly compare and examine if mistakes exist>,\n"
        "  \"has_mistakes\": <true if you found a mistake, false otherwise>,\n"
        "  \"mistakes\": [\n"
        "    {{\n"
        "      \"type\": <either INCORRECT or NO_HELP>,\n"
        "      \"output_quote\": <brief: wrong words quoted from the OUTPUT>,\n"
        "      \"sentence_quote\": <brief: correct words quoted from the SENTENCE> # ignore \"sentence_quote\" if type is NO_HELP\n"
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


# # For testing purposes
# art = {'created_en': '2024-04-06 20:32:00', 'url': 'https://en.wikipedia.org/wiki/Donal_Roche', 'title': 'Donal Roche', 'revision_id': 1218909387, 'earliest_creation_date': '2024-03-05 11:12:00', 'content': 'Donal Roche (born 16 April 1958) is an Irish Roman Catholic priest who was appointed auxiliary bishop-elect of Dublin and titular bishop of Cell Ausaille on 5 March 2024.\n\n\n== Early life and education ==\nRoche was born in Drimnagh, Dublin on 16 April 1958, one of seven children to Joe Roche and his wife Sheila. He attended both primary and secondary school at Drimnagh Castle CBS.Roche worked for four years as a clerical officer in Dublin County Council, before entering Holy Cross College in 1980 to study for the priesthood. He subsequently completed a Bachelor of Theology in St Patrick\'s College, Maynooth.Roche was ordained a priest for the Archdiocese of Dublin on 22 June 1986.\n\n\n== Presbyteral ministry ==\nFollowing ordination, Roche\'s first diocesan assignment was as a priest-teacher in Coláiste Dhúlaigh, Coolock. He was appointed diocesan advisor for religious education in primary schools in 1992, before his appointment five years later as chaplain to St Mark\'s Community School, Tallaght. During this period, Roche also spent six years as assistant diocesan vocations director.His first pastoral assignment was as curate in Lucan South parish, Lucan in 2005, before being appointed co-parish priest in Lucan South the following year.Roche was appointed administrator in Wicklow and Rathnew in 2012, during which time he was also appointed administrator in Kilbride, Barndarrig and Brittas Bay for a four-year period. In an interview with The Irish Catholic in 2022, he stated that while the local community had been "very welcoming" to newcomers, there was concern and anxiety over the recent influx of asylum seekers into Wicklow.Roche was subsequently appointed parish priest of Wicklow, Kilbride and Barndarrig in 2022. In another interview with The Irish Catholic in 2023 following the arrival of a large number of asylum seekers in Wicklow, he insisted that in spite of "understandable" concerns about housing and homelessness, Irish people were still welcoming towards asylum seekers, particularly when meeting them face-to-face.Roche was appointed moderator in Cabinteely and Johnstown-Killiney in 2023, with additional responsibility for the developing suburb of Cherrywood.In addition to his pastoral assignments, Roche has served as secretary to the diocesan council of priests since 2011. He was appointed episcopal vicar in 2019, with responsibility for the deaneries of Donnybrook, Dún Laoghaire, Bray and Wicklow, before being subsequently appointed diocesan vicar general in 2021. Roche is also a fluent Irish speaker, who has celebrated Mass and administered the sacraments in Irish throughout the Archdiocese of Dublin.\n\n\n== Episcopal ministry ==\nRoche was appointed auxiliary bishop-elect of Dublin and titular bishop of Cell Ausaille by Pope Francis on 5 March 2024. His appointment will involve supporting the Archbishop of Dublin, Dermot Farrell, in his role of leading the archdiocese through the synod on synodality. Following his appointment, Roche commended the commitment and faith of both priests and laity in the archdiocese in "challenging times".In an interview with The Irish Catholic in March 2024, he opined that the Catholic Church must do more to reach out to young people suffering from anxiety, adding that the Easter message of Christ\'s victory over death is an opportunity to bring them a message of hope in a "much more secular culture" that was being greatly influenced by social media.Roche will be consecrated on 26 May in St Andrew\'s Church, Westland Row, Dublin.\n\n\n== References ==\n\n\n== External links ==\nFather Donal Roche on Catholic-Hierarchy.org\nBishop-elect Donal Roche on GCatholic', 'passage_data': [{'start': 1007, 'end': 1095, 'contains_article': False, 'earliest_access_date': '2024-04-06 00:00:00', 'earliest_archive_date': None, 'earliest_date': '2024-03-13 00:00:00', 'references': [{'key_count': 9, 'ref_label': 'Cite web', 'access_date': '2024-04-06 00:00:00', 'date': '2024-03-13 00:00:00', 'archive_date': None}]}, {'start': 2958, 'end': 3094, 'contains_article': False, 'earliest_access_date': '2024-04-06 00:00:00', 'earliest_archive_date': None, 'earliest_date': '2024-03-07 00:00:00', 'references': [{'key_count': 9, 'ref_label': 'Cite web', 'access_date': '2024-04-06 00:00:00', 'date': '2024-03-07 00:00:00', 'archive_date': None}]}, {'start': 3094, 'end': 3445, 'contains_article': False, 'earliest_access_date': '2024-04-06 00:00:00', 'earliest_archive_date': None, 'earliest_date': '2024-03-28 00:00:00', 'references': [{'key_count': 9, 'ref_label': 'Cite web', 'access_date': '2024-04-06 00:00:00', 'date': '2024-03-28 00:00:00', 'archive_date': None}]}], 'removed_duplicates': False, 'backlinks': 3}
# classify(
#     title=art["title"],
#     section_before_passage="Following ordination, Roche's first diocesan assignment was as a priest-teacher in Coláiste Dhúlaigh, Coolock. He was appointed diocesan advisor for religious education in primary schools in 1992, before his appointment five years later as chaplain to St Mark's Community School, Tallaght.",
#     passage_text="During this period, Roche also spent six years as assistant diocesan vocations director.",
#     question="Donal Roche has once been an assistant diocesan vocations director. How many years did he spent in this position?",
#     answer_quote="six",
#     llm_output="Donal Roche has been in this position for six and a half years."
#     # llm_output="Donal Roche has been in this position for more than six years."
#     # llm_output="Donal Roche has been in this position for six years."
#     # llm_output="Donal Roche has been in this position for 6 years."
# )