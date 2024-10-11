import os
import pickle
import pandas as pd
from pprint import pprint

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "HalluRAG")

data = {}
data_struct = {}

def get_shd_prediction(answerable, pred):
    if pred is None:
        return None

    conflicting_fail_content = pred["conflicting_fail_content"]
    conflicting_fail = pred["conflicting_fail"]
    grounded_fail_content = pred["grounded_fail_content"]
    grounded_fail = pred["grounded_fail"]
    no_clear_answer_fail_content = pred["no_clear_answer_fail_content"]
    no_clear_answer_fail = pred["no_clear_answer_fail"]

    has_fail = conflicting_fail_content or conflicting_fail or grounded_fail_content or grounded_fail or no_clear_answer_fail_content or no_clear_answer_fail
    if has_fail:
        return None

    conflicting = pred["conflicting"]
    grounded = pred["grounded"]
    has_factual_information = pred["has_factual_information"]
    no_clear_answer = pred["no_clear_answer"]

    if conflicting is None or grounded is None or has_factual_information is None or no_clear_answer is None:
        return None

    if grounded and has_factual_information and no_clear_answer:
        return None

    if answerable:
        if conflicting:
            prediction = 1
        elif has_factual_information and grounded:
            prediction = 0
        elif no_clear_answer:
            prediction = 1
        else:
            if has_factual_information:
                if grounded:
                    prediction = 0
                else:
                    prediction = 1
            else:
                prediction = 0
    else:
        if conflicting:
            prediction = 1
        elif not grounded and has_factual_information:
            prediction = 1
        elif no_clear_answer:
            prediction = 0
        else:
            if has_factual_information:
                if grounded:
                    print("-"*10)
                    pprint(pred["llm_eval"])
                    print("-"*10)
                    return None
                    # raise Exception("Unanswerable question has been answered. Should not be possible!")
                else:
                    prediction = 1
            else:
                prediction = 0
    return prediction

for filename in os.listdir(DATA_DIR):
    with open(os.path.join(DATA_DIR, filename), "rb") as handle:
        d = pickle.load(handle)
        typ, rest = filename.split("_", 1)
        if rest not in data:
            data[rest] = []
        if rest not in data_struct:
            data_struct[rest] = {}
        sents = 0
        for p in d:
            for asd in p["sentence_data"]:
                # pprint(asd["pred"])
                # print(p["prompt"]["passage"]["question"])
                # print(p["prompt"]["passage"]["answer_quote"])
                # print(p["llm_response"])
                # print(p["prompt"]["answerable"])
                # print(get_shd_prediction(p["prompt"]["answerable"], asd["pred"]))
                # exit()
                if get_shd_prediction(p["prompt"]["answerable"], asd["pred"]) == None:
                    sents += 1
            # sents += len(p["sentence_data"])
        data_struct[rest][typ] = sents
        # s = d[0]
        # for i in range(len(s["sentence_data"])):
        #     for a in s["sentence_data"][i]["internal_states"]:
        #         s["sentence_data"][i]["internal_states"][a] = []
        # import json
        # with open("testtest.json", "w") as file:
        #     json.dump(s, file, ensure_ascii=False, indent=4)
        # print(s)
        # exit()
        data[rest] += d

"""
{
    "model": llm.name,
    "quantization": llm.quantization,
    "prompt": {
        "answerable": True,
        "answer_chunk_index": answer_chunk_index,
        "chunk_size": chunk_size,
        "chunks_per_prompt": chunks_per_prompt,
        "uglified": uglify_bool,
        "prompt_template_name": prompt_template_name,
        "passage": row.to_dict(),
        "other_passages": [other_chunk_row.to_dict() for _, other_chunk_row in other_chunks_df.iterrows()],
        "rag_prompt": prompt_template_function(chunks_for_answerable, row["question"])
    },
    "sentence_data": [{
        "target": None, # This is determined later using SHD and IDKC
        "cum_sentence": llm_output,
        "internal_states": internal_states,
        "pred": {a lot of bools here}
    }],
    "llm_response": llm_response
}
"""

model_data = {}
model_chunk_size = {}
model_chunks = {}
model_chunk_per_template = {}
model_chunk_answerable = {}
model_chunk_index = {}

model_chunk_per_template_answerable = {}
model_chunk_per_template_unanswerable = {}

for file in data:
    model = file.replace(".pickle", "").split("_", 1)[-1]

    predictions = []
    predictions_per_template = {
        "template_langchain_hub": [],
        "template_1": [],
        "template_2": [],
    }
    predictions_per_template_answerable = {
        "template_langchain_hub": [],
        "template_1": [],
        "template_2": [],
    }
    predictions_per_template_unanswerable = {
        "template_langchain_hub": [],
        "template_1": [],
        "template_2": [],
    }
    predictions_chunk_size = {
        "350": [],
        "550": [],
        "750": [],
    }
    predictions_chunks = {
        "1": [],
        "3": [],
        "5": [],
    }
    predictions_answerable = {
        "True": [],
        "False": [],
    }
    predictions_chunk_index = {
        "None": [],
        "0": [],
        "1": [],
        "2": [],
        "3": [],
        "4": [],
    }

    for d in data[file]:
        for sent in d["sentence_data"]:
            target = get_shd_prediction(d["prompt"]["answerable"], sent["pred"])
            predictions.append(target)
            predictions_per_template[d["prompt"]["prompt_template_name"]].append(target)
            predictions_answerable[str(d["prompt"]["answerable"])].append(target)
            predictions_chunk_size[str(d["prompt"]["chunk_size"])].append(target)
            predictions_chunks[str(d["prompt"]["chunks_per_prompt"])].append(target)

            if d["prompt"]["answerable"]:
                predictions_per_template_answerable[d["prompt"]["prompt_template_name"]].append(target)
            else:
                predictions_per_template_unanswerable[d["prompt"]["prompt_template_name"]].append(target)

            if d["prompt"]["chunks_per_prompt"] == 5:
                predictions_chunk_index[str(d["prompt"]["answer_chunk_index"])].append(target)
    
    predictions = pd.Series(predictions)

    for key in predictions_per_template:
        predictions_per_template[key] = pd.Series(predictions_per_template[key])
    for key in predictions_per_template_answerable:
        predictions_per_template_answerable[key] = pd.Series(predictions_per_template_answerable[key])
    for key in predictions_per_template_unanswerable:
        predictions_per_template_unanswerable[key] = pd.Series(predictions_per_template_unanswerable[key])
    for key in predictions_chunk_size:
        predictions_chunk_size[key] = pd.Series(predictions_chunk_size[key])
    for key in predictions_chunks:
        predictions_chunks[key] = pd.Series(predictions_chunks[key])
    for key in predictions_answerable:
        predictions_answerable[key] = pd.Series(predictions_answerable[key])
    for key in predictions_chunk_index:
        predictions_chunk_index[key] = pd.Series(predictions_chunk_index[key])

    model_data[model] = {
        "sentences": len(predictions),
        "valid": len(predictions.dropna()),
        "valid%": str(round(len(predictions.dropna()) / len(predictions) * 100, 2)) + "%",
        "hallucination": (predictions.dropna() == True).sum(),
        "hallucination_rate": str(round((predictions.dropna() == True).sum() / len(predictions.dropna()) * 100, 2)) + "%"
    }

    model_chunk_per_template[model] = {
        "template_langchain_hub": str(round((predictions_per_template["template_langchain_hub"].dropna() == True).mean() * 100, 2)) + "%",
        "template_1": str(round((predictions_per_template["template_1"].dropna() == True).mean() * 100, 2)) + "%",
        "template_2": str(round((predictions_per_template["template_2"].dropna() == True).mean() * 100, 2)) + "%",
    }

    model_chunk_per_template_answerable[model] = {
        "template_langchain_hub": str(round((predictions_per_template_answerable["template_langchain_hub"].dropna() == True).mean() * 100, 2)) + "%",
        "template_1": str(round((predictions_per_template_answerable["template_1"].dropna() == True).mean() * 100, 2)) + "%",
        "template_2": str(round((predictions_per_template_answerable["template_2"].dropna() == True).mean() * 100, 2)) + "%",
    }

    model_chunk_per_template_unanswerable[model] = {
        "template_langchain_hub": str(round((predictions_per_template_unanswerable["template_langchain_hub"].dropna() == True).mean() * 100, 2)) + "%",
        "template_1": str(round((predictions_per_template_unanswerable["template_1"].dropna() == True).mean() * 100, 2)) + "%",
        "template_2": str(round((predictions_per_template_unanswerable["template_2"].dropna() == True).mean() * 100, 2)) + "%",
    }

    model_chunk_answerable[model] = {
        "False": str(round((predictions_answerable["False"].dropna() == True).mean() * 100, 2)) + "%",
        "True": str(round((predictions_answerable["True"].dropna() == True).mean() * 100, 2)) + "%"
    }
    model_chunks[model] = {
        "1": str(round((predictions_chunks["1"].dropna() == True).mean() * 100, 2)) + "%",
        "3": str(round((predictions_chunks["3"].dropna() == True).mean() * 100, 2)) + "%",
        "5": str(round((predictions_chunks["5"].dropna() == True).mean() * 100, 2)) + "%"
    }
    model_chunk_size[model] = {
        "350": str(round((predictions_chunk_size["350"].dropna() == True).mean() * 100, 2)) + "%",
        "550": str(round((predictions_chunk_size["550"].dropna() == True).mean() * 100, 2)) + "%",
        "750": str(round((predictions_chunk_size["750"].dropna() == True).mean() * 100, 2)) + "%"
    }
    model_chunk_index[model] = {
        "None": str(round((predictions_chunk_index["None"].dropna() == True).mean() * 100, 2)) + "%",
        "0": str(round((predictions_chunk_index["0"].dropna() == True).mean() * 100, 2)) + "%",
        "1": str(round((predictions_chunk_index["1"].dropna() == True).mean() * 100, 2)) + "%",
        "2": str(round((predictions_chunk_index["2"].dropna() == True).mean() * 100, 2)) + "%",
        "3": str(round((predictions_chunk_index["3"].dropna() == True).mean() * 100, 2)) + "%",
        "4": str(round((predictions_chunk_index["4"].dropna() == True).mean() * 100, 2)) + "%"
    }


print("Info:")
print((pd.DataFrame(model_data).T).sort_index())
print("\n\nChunk Sizes:")
print((pd.DataFrame(model_chunk_size).T).sort_index())
print("\n\nNumber of Chunks:")
print((pd.DataFrame(model_chunks).T).sort_index())
print("\n\nPrompt Templates:")
print((pd.DataFrame(model_chunk_per_template).T).sort_index())
print("\n\nAnswerability:")
print((pd.DataFrame(model_chunk_answerable).T).sort_index())
print("\n\nChunk Index:")
print((pd.DataFrame(model_chunk_index).T).sort_index())

print("\n\nPrompt Templates (Answerable Only):")
print((pd.DataFrame(model_chunk_per_template_answerable).T).sort_index())
print("\n\nPrompt Templates (Unanswerable Only):")
print((pd.DataFrame(model_chunk_per_template_unanswerable).T).sort_index())

print((pd.DataFrame(data_struct).T).sort_index())

"""

Info:
                                  sentences valid  valid% hallucination hallucination_rate
Llama-2-13b-chat-hf (float8)           2188  1963  89.72%           304             15.49%
Llama-2-13b-chat-hf (int4)             2153  1941  90.15%           300             15.46%
Llama-2-13b-chat-hf (int8)             2199  1991  90.54%           315             15.82%
Llama-2-7b-chat-hf                     2201  1932  87.78%           407             21.07%
Llama-2-7b-chat-hf (float8)            2223  1989  89.47%           429             21.57%
Llama-2-7b-chat-hf (int4)              3577  3282  91.75%           695             21.18%
Llama-2-7b-chat-hf (int8)              2176  1930  88.69%           410             21.24%
Mistral-7B-Instruct-v0.1               1193  1152  96.56%           118             10.24%
Mistral-7B-Instruct-v0.1 (float8)      1202  1158  96.34%           106              9.15%
Mistral-7B-Instruct-v0.1 (int4)        1232  1205  97.81%           138             11.45%
Mistral-7B-Instruct-v0.1 (int8)        1212  1188  98.02%           129             10.86%


Chunk Sizes:
                                      350     550     750
Llama-2-13b-chat-hf (float8)       13.33%  14.42%  18.53%
Llama-2-13b-chat-hf (int4)         15.26%  15.44%  15.67%
Llama-2-13b-chat-hf (int8)         14.42%   14.2%  18.76%
Llama-2-7b-chat-hf                  20.0%  21.54%  21.75%
Llama-2-7b-chat-hf (float8)        19.68%  21.61%  23.53%
Llama-2-7b-chat-hf (int4)          20.79%  21.46%   21.3%
Llama-2-7b-chat-hf (int8)          18.66%  23.08%  22.08%
Mistral-7B-Instruct-v0.1            8.66%   9.92%  12.17%
Mistral-7B-Instruct-v0.1 (float8)   8.27%   8.31%  10.88%
Mistral-7B-Instruct-v0.1 (int4)     8.79%   9.98%  15.52%
Mistral-7B-Instruct-v0.1 (int8)      9.3%  11.59%   11.7%


Number of Chunks:
                                        1       3       5
Llama-2-13b-chat-hf (float8)        7.66%  17.87%  21.01%
Llama-2-13b-chat-hf (int4)         12.01%  16.43%  17.44%
Llama-2-13b-chat-hf (int8)          7.56%  18.58%  21.38%
Llama-2-7b-chat-hf                 14.13%  22.86%  25.95%
Llama-2-7b-chat-hf (float8)         15.5%  23.05%  26.04%
Llama-2-7b-chat-hf (int4)          16.02%  22.34%  24.32%
Llama-2-7b-chat-hf (int8)          14.33%  24.61%  24.77%
Mistral-7B-Instruct-v0.1             8.9%   8.97%  12.79%
Mistral-7B-Instruct-v0.1 (float8)   7.89%   8.04%  11.36%
Mistral-7B-Instruct-v0.1 (int4)    10.53%   8.55%   15.0%
Mistral-7B-Instruct-v0.1 (int8)      9.4%  10.94%  12.11%


Prompt Templates:
                                  template_langchain_hub template_1 template_2
Llama-2-13b-chat-hf (float8)                      21.29%      9.49%     15.25%
Llama-2-13b-chat-hf (int4)                        16.35%     10.08%     20.36%
Llama-2-13b-chat-hf (int8)                         23.7%      9.77%     13.21%
Llama-2-7b-chat-hf                                30.92%     17.76%     16.35%
Llama-2-7b-chat-hf (float8)                       32.46%     17.53%     16.72%
Llama-2-7b-chat-hf (int4)                         40.92%     16.72%     13.27%
Llama-2-7b-chat-hf (int8)                          32.0%     18.39%     15.22%
Mistral-7B-Instruct-v0.1                          14.96%      9.19%      6.67%
Mistral-7B-Instruct-v0.1 (float8)                 15.42%      8.44%      3.59%
Mistral-7B-Instruct-v0.1 (int4)                    16.8%      9.63%      8.23%
Mistral-7B-Instruct-v0.1 (int8)                   15.21%     10.33%       7.2%


Answerability:
                                    False    True
Llama-2-13b-chat-hf (float8)       12.03%  19.58%
Llama-2-13b-chat-hf (int4)          14.9%  16.27%
Llama-2-13b-chat-hf (int8)         10.73%  21.76%
Llama-2-7b-chat-hf                 14.94%  30.64%
Llama-2-7b-chat-hf (float8)        17.62%  27.92%
Llama-2-7b-chat-hf (int4)           8.36%  37.82%
Llama-2-7b-chat-hf (int8)          16.24%  28.95%
Mistral-7B-Instruct-v0.1            9.04%  11.48%
Mistral-7B-Instruct-v0.1 (float8)   7.63%  10.74%
Mistral-7B-Instruct-v0.1 (int4)     13.2%   9.68%
Mistral-7B-Instruct-v0.1 (int8)    10.22%  11.51%


Chunk Index:
                                     None       0       1       2       3       4
Llama-2-13b-chat-hf (float8)       12.16%   8.51%  26.67%  47.54%  43.33%  28.21%
Llama-2-13b-chat-hf (int4)         14.81%   6.98%  14.93%  20.63%   37.5%  20.29%
Llama-2-13b-chat-hf (int8)         11.02%  14.75%   30.0%  45.16%  48.39%  28.75%
Llama-2-7b-chat-hf                 17.02%   20.0%  46.94%   40.0%  43.64%  40.62%
Llama-2-7b-chat-hf (float8)        21.83%  19.35%  36.73%  37.04%  33.33%  34.85%
Llama-2-7b-chat-hf (int4)           9.62%  25.56%  48.48%  46.94%  53.49%  42.25%
Llama-2-7b-chat-hf (int8)           16.8%  19.67%   44.0%  41.38%  34.55%   37.7%
Mistral-7B-Instruct-v0.1            7.07%  13.16%  15.38%  34.15%  19.35%  11.36%
Mistral-7B-Instruct-v0.1 (float8)   6.22%  10.81%   7.69%  31.71%   20.0%  13.64%
Mistral-7B-Instruct-v0.1 (int4)    14.85%   9.62%  10.26%   27.5%  17.95%   12.5%
Mistral-7B-Instruct-v0.1 (int8)     6.86%  11.32%   12.5%  30.95%  15.79%  15.91%


"""