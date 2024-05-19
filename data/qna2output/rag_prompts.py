from typing import Dict, List

CHUNKS_PER_PROMPT = [1, 2, 3, 5]
CHUNK_SIZE = [-1, 250, 500]

def template_langchain_hub(chunks: List[Dict[str, str]], question: str) -> List[Dict[str, str]]:
    template = []
    context = ""

    for i, chunk in enumerate(chunks):
        context += f"### Chunk {i+1}: {chunk['title']}\n{chunk['content']}\n\n"

    # This string is taken from langchain hub via 'hub.pull("rlm/rag-prompt")' (on May 19, 2024)
    template.append({
        "role": "user",
        "content": f"You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
    })

    return template

def template_1(chunks: List[Dict[str, str]], question: str) -> List[Dict[str, str]]:
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
        "content": f"Your knowledge is limited to only this information:\n{context}\nQUESTION: {question}\nANSWER:"
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
        context += f"### {chunk['title']} ###\n{chunk['content']}\n\n"

    template.append({
        "role": "user",
        "content": f"Only use the information included in these chunks to answer the question:\n{context}\nQUESTION: {question}\nREMINDER: If no chunk contains the information asked for, briefly explain that you cannot answer the question.\nRESPONSE:"
    })

    return template
            
PROMPT_TEMPLATES = [template_langchain_hub, template_1, template_2]