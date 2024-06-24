from typing import Dict, List
import random

CHUNKS_PER_PROMPT = [1, 3, 5]
CHUNK_SIZE = [350, 550, 750]
UGLIFY = [False] # True
RANDOM_STATE = 432

random.seed(RANDOM_STATE)

def template_langchain_hub(chunks: List[Dict[str, str]], question: str) -> List[Dict[str, str]]:
    template = []
    context = ""

    for i, chunk in enumerate(chunks):
        context += f"### Chunk {i+1}: {chunk['title']}\n{chunk['content']}\n\n"

    # This string is taken from langchain hub via 'hub.pull("rlm/rag-prompt")' (on May 19, 2024)
    # But adjusted: replaced "three sentences at maximum" with "as few sentences as possible"
    template.append({
        "role": "user",
        "content": f"You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use as few sentences as possible and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
    })

    return template

def template_1(chunks: List[Dict[str, str]], question: str) -> List[Dict[str, str]]:
    template = []

    # The system message is inspired by (Hicke et. al, 2023) -> https://arxiv.org/html/2311.02775v3
    template.append({
        "role": "system",
        "content": "You are a helpful, respectful, and honest assistant for a question-answering task. You are provided pieces of context that MIGHT contain the answer to the question. Your concise answer should solely be based on these pieces. Always answer as helpfully as possible, while being safe. Your answer should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, state that so you don't share false information. Do not refer to chunks literally. Do not use the word 'chunk', just use their information for your answer. Do NOT start with 'Based on...'"
    })
    context = ""

    for i, chunk in enumerate(chunks):
        context += f"### Chunk {i+1}: {chunk['title']}\n{chunk['content']}\n\n"

    template.append({
        "role": "user",
        "content": f"Your knowledge is limited to only this information:\n{context}\nQUESTION: {question}\nBRIEF ANSWER:"
    })

    return template
            
def template_2(chunks: List[Dict[str, str]], question: str) -> List[Dict[str, str]]:
    template = []

    # The system message is inspired by (Hicke et. al, 2023) -> https://arxiv.org/html/2311.02775v3
    template.append({
        "role": "system",
        "content": "You are a helpful, respectful, and honest assistant for a question-answering task. You are provided pieces of context that MIGHT contain the answer to the question. Your concise answer should solely be based on these pieces. Always answer as helpfully as possible, while being safe. Your answer should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, state that so you don't share false information."
    })
    context = ""

    for i, chunk in enumerate(chunks):
        context += f"### Chunk {i+1}: {chunk['title']}\n{chunk['content']}\n\n"

    template.append({
        "role": "user",
        "content": f"Only use the information included in these chunks to answer the question:\n{context}\nQUESTION: {question}\nREMINDER: If no chunk contains the information asked for, briefly explain that you cannot answer the question. However, do not refer to chunks literally. Do not use the word 'chunk' or that chunks were provided to you, just use their information to answer the QUESTION. Do NOT start with 'Based on...'\nBRIEF RESPONSE:"
    })

    return template

def uglify(text: str) -> str:
    """
    Make a given string "ugly" but still somewhat readable by inserting random
    amounts of spaces and newlines, and occasionally removing or swapping characters or words.

    Args:
        text (str): The input string to be uglified.

    Returns:
        str: The uglified version of the input string.
    """

    words = text.split()
    ugly_text = []

    for word in words:
        # Randomly decide to drop a word (3% chance)
        if random.random() < 0.03:
            continue
        
        # Randomly decide to drop a character (5% chance)
        if random.random() < 0.05:
            char_idx = random.randint(0, len(word) - 1)
            word = word[:char_idx] + word[char_idx + 1:]
        
        # Randomly decide to swap two characters (4% chance)
        if len(word) > 1 and random.random() < 0.04:
            idx1, idx2 = random.sample(range(len(word)), 2)
            word = list(word)
            word[idx1], word[idx2] = word[idx2], word[idx1]
            word = ''.join(word)

        ugly_text.append(word)
        
        # Randomly add spaces (5% chance)
        if random.random() < 0.05:
            spaces = ' ' * random.randint(1, 4)
            ugly_text.append(spaces)

        # Randomly add newlines (2% chance)
        if random.random() < 0.02:
            newlines = '\n' * random.randint(1, 2)
            ugly_text.append(newlines)

    return ' '.join(ugly_text)

PROMPT_TEMPLATES = [template_langchain_hub, template_1, template_2]