from tools.command_generator import generate_prompts, prefix_prompt
from tools.evaluate_response import get_replacements
from tqdm import tqdm
from textblob import TextBlob
from termcolor import colored
import pandas as pd
import numpy as np
def run_prompts_blob(model, sentence, anchor_idx, prompts_per_word=20):
    prompts = generate_prompts(sentence, anchor_idx)
    all_replacements = []
    for prompt in prompts:
        replacements = []
        for _ in tqdm(
            range(prompts_per_word),
            desc=f"Input: {prompt}",
        ):
            response = model.get_response(
                prefix_prompt(prompt),
            ).strip()
            if response:
                replacement = get_replacements(prompt, response)
                if replacement:
                    replacements.append(replacement)
        if len(replacements) > 0:
            all_replacements.append(replacements)
    return all_replacements



def colorize_sentence(sentence, saliency):
    words = sentence.split()
    for i, word in enumerate(words):
        saliency_value = saliency[i]
        if saliency_value > 0.5:
            words[i] = colored(words[i], 'red')
        elif saliency_value > 0.3:
            words[i] = colored(words[i], 'yellow')
        else:
            words[i] = colored(words[i], 'green')
    return ' '.join(words)

def calculate_pmis(sentence, all_responses, anchor_word_idx, prompts_per_word):
    words = sentence.split()
    words[-1] = words[-1].replace(".", "")
    words = [word.lower() for word in words]

    p_df = pd.DataFrame(columns=words)
    p_df.loc['px'] = 0
    p_df.loc['py'] = 0
    p_df.loc['pxy'] = 0
    p_df.loc['pmi'] = 0
    p_df.loc['saliency'] = 0

    idx_y = 0
    word_x = words[anchor_word_idx].lower()

    for i, responses in enumerate(all_responses):
        px = 1e-10
        py = 1e-10
        pxy = 1e-10

        if anchor_word_idx == i:
            idx_y = 1

        word_y = words[i+idx_y].lower()

        for response in responses:
            x = response[0]
            y = response[1]
            if x == word_x:
                px+=1
            if y == word_y:
                py+=1
            if x == word_x and y == word_y:
                pxy+=1

        px = px/prompts_per_word
        py = py/prompts_per_word
        pxy = pxy/prompts_per_word
        pmi = np.log2(pxy/(px*py))

        p_df.at['px', word_y] = px
        p_df.at['py', word_y] = py
        p_df.at['pxy', word_y] = pxy
        p_df.at['pmi', word_y] = pmi

    p_df[word_x] = np.nan
    min_pmi = np.round(p_df.loc['pmi'].min(),10)
    max_pmi = p_df.loc['pmi'].max()
    p_df.loc['saliency'] = (p_df.loc['pmi']-min_pmi)/(max_pmi-min_pmi)

    return p_df