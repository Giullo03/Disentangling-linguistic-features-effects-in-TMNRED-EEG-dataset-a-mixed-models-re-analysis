import stanza
from transformers import BertTokenizer, GPT2LMHeadModel # Changed GPT2Tokenizer to BertTokenizer
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# stanza.download("zh")    # only needed once

# the following need to stay out of any loop
nlp = stanza.Pipeline("zh", processors="tokenize,pos,lemma,depparse")
gpt_tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall") # Changed GPT2Tokenizer to BertTokenizer
gpt_model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
gpt_model.eval()
sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

fp = Path(__file__).parent / "strokeCount.json"   # file created based off the unihan strokes dataset of 2025-08-18 10:51 8.1M
with open(fp, encoding="utf-8") as f:
    data = json.load(f)

char_freq_path = Path(__file__).parent / "SUBTLEX-CH-CHR.xlsx"
char_freq_df = pd.read_excel(char_freq_path, skiprows=2)
char_freq_df = char_freq_df[["Character", "logCHR"]]
char_freq_dict = dict(zip(char_freq_df["Character"], char_freq_df["logCHR"]))

word_freq_path = Path(__file__).parent / "SUBTLEX-CH-WF.xlsx"
word_freq_df = pd.read_excel(word_freq_path, skiprows=2)
word_freq_df = word_freq_df[["Word", "logW"]]
word_freq_dict = dict(zip(word_freq_df["Word"], word_freq_df["logW"]))


#-------SYNTACTIC COMPLEXITY (from dependency tree depth)-------#
def syntactic_complexity(doc):
    sent = doc.sentences[0]

    tree = {}
    for word in sent.words:
        tree.setdefault(word.head, []).append(word.id)

    def depth(node):
        if node not in tree:
            return 1
        return 1 + max(depth(child) for child in tree[node])

    return depth(0) 


#-------SURPRISAL-------#
def mean_surprisal(sentence):

    doc = nlp(sentence)
    stanza_words = [word.text for sent in doc.sentences for word in sent.words]

    inputs = gpt_tokenizer(sentence, return_tensors="pt")

    if inputs["input_ids"].shape[1] < 2:
        return np.nan

    tokens = gpt_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = gpt_model(**inputs)

    logits = outputs.logits

    subtoken_surprisal = []
    subtoken_strings = []

    for i in range(1, inputs["input_ids"].shape[1]):
        probs = F.softmax(logits[0, i-1], dim=-1)
        true_token_id = inputs["input_ids"][0, i]
        surprisal = -torch.log(probs[true_token_id]).item()

        subtoken_surprisal.append(surprisal)
        subtoken_strings.append(tokens[i])

    word_surprisals = []
    current_word = ""
    current_sum = 0
    stanza_index = 0

    for token, s in zip(subtoken_strings, subtoken_surprisal):
        cleaned_token = token.replace("##", "")
        current_word += cleaned_token
        current_sum += s

        if stanza_index < len(stanza_words) and current_word == stanza_words[stanza_index]:
            word_surprisals.append(current_sum)
            stanza_index += 1
            current_word = ""
            current_sum = 0

    if len(word_surprisals) == 0:
        return np.nan

    return np.mean(word_surprisals)


#-------STROKES-------#
def mean_strokes(sentence):
    chars = [c for c in sentence if '\u4e00' <= c <= '\u9fff']
    stroke_values = [data.get(c, 0) for c in chars]

    if len(stroke_values) == 0:
        return np.nan

    return np.mean(stroke_values)



#-------FREQUENCY-------#
# based off character frequency
def mean_char_frequency(sentence):
    chars = [c for c in sentence if '\u4e00' <= c <= '\u9fff']
    values = [char_freq_dict.get(c, np.nan) for c in chars]
    values = [v for v in values if not np.isnan(v)]

    if len(values) == 0:
        return np.nan

    return np.mean(values)

# based off word frequency
def mean_word_frequency(doc):
    stanza_words = [word.text for sent in doc.sentences for word in sent.words]
    values = [word_freq_dict.get(w, np.nan) for w in stanza_words]
    values = [v for v in values if not np.isnan(v)]

    if len(values) == 0:
        return np.nan

    return np.mean(values)


#-------COMPUTE NLP FEATURES-------#
file_path = Path(__file__).parent / "dataset-name"
xls = pd.ExcelFile(file_path)

all_sentences = []
metadata = []

for sheet_name in xls.sheet_names:
    df_block = pd.read_excel(xls, sheet_name=sheet_name)

    for i, row in df_block.iterrows():

        sentence = str(row["Material statement"])
        stimulus_type = row["Stimulus type"]

        all_sentences.append(sentence)

        metadata.append({
            "block": sheet_name,
            "stimulus_type": stimulus_type
        })


output_dir = Path("output-directory-path")
results = []
for sentence, meta in zip(all_sentences, metadata):
    doc = nlp(sentence)

    results.append({
        "block": meta["block"],
        "stimulus_type": meta["stimulus_type"],
        "syntactic_complexity": syntactic_complexity(doc),  # Syntactic Complexity
        "surprisal": mean_surprisal(sentence),  # Surprisal
        "strokes": mean_strokes(sentence),  # Strokes
        "char_frequency": mean_char_frequency(sentence),    # Frequency based off characters
        "word_frequency": mean_word_frequency(doc)  # Frequency based off words
    })
features_df = pd.DataFrame(results)
features_df.to_csv(output_dir / "nlp_sentence_features.csv", index=False)
print("NLP features saved.")


embeddings = sbert_model.encode(
    all_sentences,
    batch_size=32,
    show_progress_bar=True
)

similarity_matrix = cosine_similarity(embeddings)

sentence_ids = [
    f"{meta['block'].lower()}_{str(meta['stimulus_type']).lower().replace(' ', '')}"
    for meta in metadata
]

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=sentence_ids,
    columns=sentence_ids
)

similarity_df.to_csv(output_dir / "semantic_similarity_matrix.csv")
np.save(output_dir / "semantic_similarity_matrix.npy", similarity_matrix)

print("Similarity matrix saved.")

