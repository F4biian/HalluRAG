# HalluRAG: Detecting Hallucinations in RAG Applications Using an LLM’s Internal States

Welcome to the official repository for **HalluRAG**, a dataset for detecting sentence-level hallucinations in Retrieval-Augmented Generation (RAG) applications by leveraging the internal states of large language models (LLMs). This repository contains the implementation of methods described in our paper, "HalluRAG: Detecting Hallucinations in RAG Applications Using an LLM’s Internal States."

RAG-based systems combine external retrieval mechanisms with generative models to produce information-rich responses. However, hallucinations — generated content that is ungrounded in an LLM's knowledge — remain a critical challenge. HalluRAG introduces an approach to identifying these hallucinations by training a multilayer perceptron (MLP) on the `contextualized embedding vectors` and `intermediate activation values` within the LLM.

## Requirements

- Python version: `Python 3.10.12`  
- Install all required packages: `pip install -r requirements.txt`
- For the dataset creation:
    - A `.env` file at the repo's root directory. See [`.env.example`](.env.example) for details.
    - A GPU with sufficient specs to run an LLM. We used an *NVIDIA RTX A6000*.

## Overview

This project's core components are the dataset creation ([`data`](data/) folder) and training a classifier on this data ([`classification`](classification/) folder).

> **Note:** All big files have been excluded from this repo. They can be downloaded from here (12.18 GB):  
> https://drive.google.com/file/d/1YEkrV26TOoF1YaKg-4urqcyqSOjBibnZ
> 
> If you only want to download the final HalluRAG dataset (2.63 GB):  
> https://drive.google.com/file/d/1hkM8yygVQKXkBgOB98R8nuB0MzPAy5sp

### `data`: Creating HalluRAG

#### Step 0) [`data/wikipedia/`](data/wikipedia/)
Scraping recent Wikipedia articles using [`data/wikipedia/wikipedia_scraper.py`](data/wikipedia/wikipedia_scraper.py). An article could look like this (abbreviated and taken from `articles_2024-03-21.json`):
<details>
<summary>Click to reveal hidden content</summary>  

```json
{
    "created_en": "2024-03-21 23:50:00",
    "url": "https://en.wikipedia.org/wiki/Naydeen_Gonz%C3%A1lez-De_Jes%C3%BAs",
    "title": "Naydeen González-De Jesús",
    "revision_id": 1214987545,
    "earliest_creation_date": null,
    "content": "Naydeen González-De Jesús is an American academic  [...]  presidential project executive by the Alamo Colleges District.\n\n\n== References ==",
    "passage_data": [
        {
            "start": 158,
            "end": 252,
            "contains_article": true,
            "earliest_access_date": "2024-03-21 00:00:00",
            "earliest_archive_date": null,
            "earliest_date": "2022-12-14 00:00:00",
            "references": [
                {
                    "key_count": 9,
                    "ref_label": "Cite web",
                    "access_date": "2024-03-21 00:00:00",
                    "date": "2022-12-14 00:00:00",
                    "archive_date": null
                }
            ]
        },
        ...
    ],
    "removed_duplicates": false,
    "backlinks": 1
}
```
</details>

#### Step 1) [`data/wikipedia/`](data/wikipedia2qna/)
Turning passages from Wikipedia articles to Q&A pairs using [`data/wikipedia2qna/2qna.py`](data/wikipedia2qna/2qna.py). A Q&A pair could look like this (taken from `qna_per_passage.json`):
<details>
<summary>Click to reveal hidden content</summary>  

```json
{
    "useful_art_i": 2,
    "useful_passage_i": 0,
    "article_title": "News Now",
    "passage_start": 180,
    "passage_end": 242,
    "context": "News Now or just Now is an upcoming Portuguese news channel, owned by the Medialivre group. It will be a news channel, competing against SIC Notícias and CNN Portugal for viewers.",
    "passage_text": "The channel is scheduled to launch in 2024 by the end of June.",
    "question": "When is the News Now channel expected to launch?",
    "answer_quote": "2024 by the end of June"
}
```
</details>  
<br>

In order to generate such Q&A pairs, access to the OpenAI API is required. Inserting your API key into the `.env` file is already sufficient.

#### Step 2) [`data/qna2output/`](data/qna2output/)
Creating (un)answerable RAG prompts (stored in [`data/qna2output/rag_prompts.py`](data/qna2output/rag_prompts.py)), passing them to *LLaMA 2 7B Chat HF*, *LLaMA 2 13B Chat HF*, and *Mistral 7B Instruct v0.1*, and retrieving the internal states using [`data/qna2output/2output.py`](data/qna2output/2output.py). An entry might look like this (abbreviated):
<details>
<summary>Click to reveal hidden content</summary>  

```json
{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "quantization": null,
    "prompt": {
        "qna_id": "378_0",
        "useful_art_i": 378,
        "useful_passage_i": 0,
        "answerable": false,
        "answer_chunk_index": null,
        "chunk_size": 350,
        "chunks_per_prompt": 1,
        "uglified": false,
        "prompt_template_name": "template_langchain_hub",
        "passage": {
            "useful_art_i": 378,
            "useful_passage_i": 0,
            "article_title": "Statue of Elizabeth II, Oakham",
            "passage_start": 705,
            "passage_end": 793,
            "context": "A statue of Queen Elizabeth II  [...]  words \"Queen Elizabeth II, 1926–2022.",
            "passage_text": "Erected as a tribute to her late Majesty through public subscription by Rutland people\".",
            "question": "How was the statue of Queen Elizabeth II in Oakham funded?",
            "answer_quote": "through public subscription by Rutland people"
        },
        "other_passages": [{
            "useful_art_i": 1002,
            "useful_passage_i": 0,
            "article_title": "Lady Killers (G-Eazy song)",
            "passage_start": 1737,
            "passage_end": 1790,
            "context": "G-Eazy released a remix on May 2, 2024, titled \"Lady Killers III\".",
            "passage_text": "It was produced by MD$, Christoph Andersson and Tane.",
            "question": "Who produced the remix 'Lady Killers III' by G-Eazy?",
            "answer_quote": "MD$, Christoph Andersson and Tane"
        }],
        "rag_prompt": [{
            "role": "user",
            "content": "You are an assistant for question-answering  [...]  Christoph Andersson and Tane.\n== References ==\n\n \nAnswer:"
        }]
    },
    "sentence_data": [
        {
            "target": null,
            "cum_sentence": "The statue of Queen Elizabeth II in Oakham was funded through a combination of public and private sources.",
            "internal_states": {
                "layer_50_last_token": [-0.01398851815611124, ..., 0.026720266789197922],
                # -> length: 4096
                "layer_100_last_token": [0.18470054864883423, ..., -0.09891434758901596],
                # -> length: 4096
                "activations_layer_50_last_token": [0.022188670933246613, ..., -0.0044380128383636475],
                # -> length: 11008
                "activations_layer_100_last_token": [0.030604401603341103, ..., -0.09061101078987122],
                # -> length: 11008
                "probability": [8.984689338831231e-05, ..., 0.19893746078014374],
                # -> length: 23 (number of tokens of cum_sentence)
                "entropy": [0.003061262657865882, ..., 1.1841719150543213]
                # -> length: 23 (number of tokens of cum_sentence)
            }
        },
        {
            "target": null,
            "cum_sentence": "The statue of Queen Elizabeth II in Oakham was funded through a combination of public and private sources. I don't have access to specific information on the funding details, so I cannot provide a definitive answer.",
            "internal_states": {
                "layer_50_last_token": [...],
                # -> length: 4096
                "layer_100_last_token": [...],
                # -> length: 4096
                "activations_layer_50_last_token": [...],
                # -> length: 11008
                "activations_layer_100_last_token": [...],
                # -> length: 11008
                "probability": [...],
                # -> length: 47 (number of tokens of cum_sentence)
                "entropy": [...]
                # -> length: 47 (number of tokens of cum_sentence)
            }
        }
    ],
    "llm_response": "The statue of Queen Elizabeth II in Oakham was funded through a combination of public and private sources. I don't have access to specific information on the funding details, so I cannot provide a definitive answer."
}
```

</details>  
<br>

This step requires to do manual changes to the transformers library (`git+https://github.com/huggingface/transformers.git@8127f39624f587bdb04d55ab655df1753de7720a`). Instructions are given for each LLM in the associated script. For instance, instructions for *Mistral 7B Instruct v0.1* can be found at [`models/mistral.py`](models/mistral.py).

#### Step 3) [`data/supervised_hallucination_classification/`](data/supervised_hallucination_classification/)
Make GPT-4o label all sentences as either hallucinated or non-hallucinated using [`data/supervised_hallucination_classification/run_shd.py`](data/supervised_hallucination_classification/run_shd.py). Afterward, there should be a HalluRAG folder containing files with entries that look like the following:

<details>
<summary>Click to reveal hidden content</summary>  

```json
{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "quantization": null,
    "prompt": {
        "qna_id": "378_0",
        "useful_art_i": 378,
        "useful_passage_i": 0,
        "answerable": false,
        "answer_chunk_index": null,
        "chunk_size": 350,
        "chunks_per_prompt": 1,
        "uglified": false,
        "prompt_template_name": "template_langchain_hub",
        "passage": {
            "useful_art_i": 378,
            "useful_passage_i": 0,
            "article_title": "Statue of Elizabeth II, Oakham",
            "passage_start": 705,
            "passage_end": 793,
            "context": "A statue of Queen Elizabeth II by Hywel Pratley stands in Oakham, the county town of Rutland in the East Midlands of England. It was unveiled on 21 April 2024, which would have been the Queen's 98th birthday. The 7ft (2.1m) tall sculpture was commissioned by the Lord Lieutenant of Rutland and was funded via donations from businesses and members of the public, at a cost of £125,000. It is the first memorial to Elizabeth II to have been unveiled after her death in September 2022. The statue portrays the Queen in Garter robes and sash wearing the George IV State Diadem, with one royal corgi at her feet and another two on the plinth. Inscribed beneath it are the words \"Queen Elizabeth II, 1926–2022.",
            "passage_text": "Erected as a tribute to her late Majesty through public subscription by Rutland people\".",
            "question": "How was the statue of Queen Elizabeth II in Oakham funded?",
            "answer_quote": "through public subscription by Rutland people"
        },
        "other_passages": [{
            "useful_art_i": 1002,
            "useful_passage_i": 0,
            "article_title": "Lady Killers (G-Eazy song)",
            "passage_start": 1737,
            "passage_end": 1790,
            "context": "G-Eazy released a remix on May 2, 2024, titled \"Lady Killers III\".",
            "passage_text": "It was produced by MD$, Christoph Andersson and Tane.",
            "question": "Who produced the remix 'Lady Killers III' by G-Eazy?",
            "answer_quote": "MD$, Christoph Andersson and Tane"
        }],
        "rag_prompt": [{
            "role": "user",
            "content": "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use as few sentences as possible and keep the answer concise.\nQuestion: How was the statue of Queen Elizabeth II in Oakham funded? \nContext: ### Chunk 1: Lady Killers (G-Eazy song)\np-Hop Songs chart, peaking at number 47. It also debuted at number 6 on the Bubbling Under Hot 100.\n=== Charts ===\n== Lady Killers III ==\nG-Eazy released a remix on May 2, 2024, titled \"Lady Killers III\". It was produced by MD$, Christoph Andersson and Tane.\n== References ==\n\n \nAnswer:"
        }]
    },
    "sentence_data": [
        {
            "target": "hallucinated", # aka 1  ("invalid" stands for null)
            "cum_sentence": "The statue of Queen Elizabeth II in Oakham was funded through a combination of public and private sources.",
            "internal_states": {
                "layer_50_last_token": [-0.01398851815611124, ..., 0.026720266789197922],
                # -> length: 4096
                "layer_100_last_token": [0.18470054864883423, ..., -0.09891434758901596],
                # -> length: 4096
                "activations_layer_50_last_token": [0.022188670933246613, ..., -0.0044380128383636475],
                # -> length: 11008
                "activations_layer_100_last_token": [0.030604401603341103, ..., -0.09061101078987122],
                # -> length: 11008
                "probability": [8.984689338831231e-05, ..., 0.19893746078014374],
                # -> length: 23 (number of tokens of cum_sentence)
                "entropy": [0.003061262657865882, ..., 1.1841719150543213]
                # -> length: 23 (number of tokens of cum_sentence)
            },
            "pred": {
                "conflicting_fail_content": false,
                "conflicting_fail": false,
                "grounded_fail_content": false,
                "grounded_fail": false,
                "no_clear_answer_fail_content": false,
                "no_clear_answer_fail": false,
                "conflicting": true,
                "grounded": false,
                "has_factual_information": true,
                "no_clear_answer": false,
                "llm_eval": {"conflicting": {"section_content": "The statue of Queen Elizabeth II in Oakham was funded through a combination of public and private sources.", "thoughts1": "The detail 'funded through a combination of public and private sources' conflicts with the NECESSARY CHUNK, which states it was funded 'through public subscription by Rutland people'.", "thoughts2": "This section might conflict with SECTION 2, which states the AI does not have access to specific information.", "result": true, "necessary_chunk_quote": "Erected as a tribute to her late Majesty through public subscription by Rutland people", "section_quote": "The statue of Queen Elizabeth II in Oakham was funded through a combination of public and private sources."}, "grounded": {"section_content": "The statue of Queen Elizabeth II in Oakham was funded through a combination of public and private sources.", "thoughts1": "The detail 'funded through a combination of public and private sources' is not grounded in the NECESSARY CHUNK.", "thoughts2": "The section contains factual information that is not grounded in the NECESSARY CHUNK.", "has_factual_information": true, "result": false, "section_quote": "", "necessary_chunk_quote": ""}, "cannot_really_answer": {"section_content": "The statue of Queen Elizabeth II in Oakham was funded through a combination of public and private sources.", "thoughts": "The AI did not state that it cannot access the information or does not know the answer.", "result": false, "section_quote": ""}}
            }
        },
        {
            "target": "non-hallucinated", # aka 0 ("invalid" stands for null)
            "cum_sentence": "The statue of Queen Elizabeth II in Oakham was funded through a combination of public and private sources. I don't have access to specific information on the funding details, so I cannot provide a definitive answer.",
            "internal_states": {
                "layer_50_last_token": [...],
                # -> length: 4096
                "layer_100_last_token": [...],
                # -> length: 4096
                "activations_layer_50_last_token": [...],
                # -> length: 11008
                "activations_layer_100_last_token": [...],
                # -> length: 11008
                "probability": [...],
                # -> length: 47 (number of tokens of cum_sentence)
                "entropy": [...]
                # -> length: 47 (number of tokens of cum_sentence)
            },
            "pred": {
                "conflicting_fail_content": false,
                "conflicting_fail": false,
                "grounded_fail_content": false,
                "grounded_fail": false,
                "no_clear_answer_fail_content": false,
                "no_clear_answer_fail": false,
                "conflicting": false,
                "grounded": true,
                "has_factual_information": false,
                "no_clear_answer": true,
                "llm_eval": {"conflicting": {"section_content": "I don't have access to specific information on the funding details, so I cannot provide a definitive answer.", "thoughts1": "This section does not conflict with the NECESSARY CHUNK as it states the AI does not have access to specific information.", "thoughts2": "This section might conflict with SECTION 1, which provides a specific answer.", "result": false, "necessary_chunk_quote": "", "section_quote": ""}, "grounded": {"section_content": "I don't have access to specific information on the funding details, so I cannot provide a definitive answer.", "thoughts1": "The section does not provide any factual information that needs to be grounded.", "thoughts2": "The section does not contain factual information.", "has_factual_information": false, "result": true, "section_quote": "", "necessary_chunk_quote": ""}, "cannot_really_answer": {"section_content": "I don't have access to specific information on the funding details, so I cannot provide a definitive answer.", "thoughts": "The AI clearly states that it does not have access to specific information and cannot provide a definitive answer.", "result": true, "section_quote": "I don't have access to specific information on the funding details, so I cannot provide a definitive answer."}}
            }
        }
    ],
    "llm_response": "The statue of Queen Elizabeth II in Oakham was funded through a combination of public and private sources. I don't have access to specific information on the funding details, so I cannot provide a definitive answer."
}
```

</details>  
<br>


### `classification`: Training on HalluRAG

This part requires the HalluRAG dataset or another dataset with the same structure. HalluRAG can be downloaded from here:  
https://drive.google.com/file/d/1hkM8yygVQKXkBgOB98R8nuB0MzPAy5sp

You can use [`classification/hallurag_clf_train.py`](classification/hallurag_clf_train.py) to train the classifier outlined in [`classification/hallu_clf.py`](classification/hallu_clf.py) on particular internal states of a particular LLM. The results are written into a `.json` file in the same directory. Then, you run the script [`classification/analyze_results.py`](classification/analyze_results.py) on that `.json` file to obtain a table with the specified metric. The following metrics are available for each `'train'`, `'val'`, `'test'`, and `'test_random'`:  

<details>
<summary>Click to reveal hidden content</summary>  

```python
[
    'loss',
    'cohen_kappa_threshold',
    'cohen_kappa',
    'mcc_threshold',
    'mcc',
    'accuracy_threshold',
    'accuracy',
    'confusion_matrix',
    'f1_hallucinated_threshold',
    'recall_hallucinated',
    'precision_hallucinated',
    'f1_hallucinated',
    'fpr_hallucinated',
    'tpr_hallucinated',
    'roc_auc_hallucinated',
    'P_hallucinated',
    'R_hallucinated',
    'auc_pr_hallucinated',
    'f1_grounded_threshold',
    'recall_grounded',
    'precision_grounded',
    'f1_grounded',
    'fpr_grounded',
    'tpr_grounded',
    'roc_auc_grounded',
    'P_grounded',
    'R_grounded',
    'auc_pr_grounded'
]
```

</details>

## Acknowledgments

A special thank you to [Weihang Su](https://github.com/oneal2000) and [Wang Yue](https://github.com/bebr2) for their invaluable support in the extraction of internal states.

