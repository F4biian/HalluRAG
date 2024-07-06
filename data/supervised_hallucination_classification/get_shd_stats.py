import os
import pickle
import pandas as pd
from pprint import pprint

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "HalluRAG")

data = {}

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

for file in data:
    model = file.replace(".pickle", "").split("_", 1)[-1]

    predictions = []
    predictions_per_template = {
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

            if d["prompt"]["chunks_per_prompt"] == 5:
                predictions_chunk_index[str(d["prompt"]["answer_chunk_index"])].append(target)
    
    predictions = pd.Series(predictions)

    for key in predictions_per_template:
        predictions_per_template[key] = pd.Series(predictions_per_template[key])
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