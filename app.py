from flask import Flask, request, jsonify
import os
import re
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from flask_cors import CORS
import nltk
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# Load Spacy for French lemmatization
nlp = spacy.load('fr_core_news_md')

# File paths for dictionary files
EN_FR_FILE = "en-fr-enwiktionary.txt"
FR_EN_FILE = "fr-en-enwiktionary.txt"

# Initialize the lemmatizer for English
lemmatizer = WordNetLemmatizer()

# Define French articles to strip
FRENCH_ARTICLES = ("le ", "la ", "les ", "l'")

def clean_input(word):
    word = word.strip().lower()
    if word.startswith("to "):
        word = word[3:]
    for article in FRENCH_ARTICLES:
        if word.startswith(article):
            word = word[len(article):]
    return word

def lemmatize_word_english(word):
    noun_lemma = lemmatizer.lemmatize(word, wordnet.NOUN)
    verb_lemma = lemmatizer.lemmatize(word, wordnet.VERB)
    return {word, noun_lemma, verb_lemma}

def lemmatize_word_french(word):
    doc = nlp(word)
    return {word, doc[0].lemma_ if doc else word}

def extract_see_reference(line):
    see_match = re.search(r"SEE:\s*(.*?)\s*::", line, re.IGNORECASE)
    if see_match:
        return see_match.group(1).strip()
    return None

def remove_phonetics(text):
    return re.sub(r'/[^/]+/', '', text)

def remove_extra_spaces(text):
    return re.sub(r'\}\s+', '} ', text)

def load_lines(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f]
    return []

def build_index(lines):
    index = {}
    for line in lines:
        m = re.match(r"^(.*?)\s*\{", line, re.IGNORECASE)
        if m:
            key = m.group(1).strip().lower()
            index.setdefault(key, []).append(line)
    return index

EN_FR_LINES = load_lines(EN_FR_FILE)
FR_EN_LINES = load_lines(FR_EN_FILE)

en_fr_index = build_index(EN_FR_LINES)
fr_en_index = build_index(FR_EN_LINES)

def find_lines_in_index(index, search_word, visited=None):
    if visited is None:
        visited = set()
    results = []
    key = search_word.lower()
    if key in visited:
        return results
    visited.add(key)
    if key in index:
        for line in index[key]:
            see_ref = extract_see_reference(line)
            if see_ref:
                recursed = find_lines_in_index(index, see_ref, visited)
                if recursed and len(recursed) <= 2:
                    results.extend(recursed)
                elif not recursed:
                    results.append(line)
            else:
                results.append(line)
    return results

def find_lines_after_colon_from_lines(lines, search_phrase):
    matching_lines = []
    for line in lines:
        if "::" in line:
            after_text = line.split("::", 1)[1]
            if search_phrase.lower() in after_text.lower():
                matching_lines.append(line)
    return matching_lines

def translation_has_letters(line):
    if "::" in line:
        right = line.split("::", 1)[1].strip()
        return bool(re.search(r"[a-zA-Z]", right))
    return True

def translate_word(word):
    cleaned_word = clean_input(word)
    if cleaned_word in en_fr_index:
        input_lang = "EN"
    elif cleaned_word in fr_en_index:
        input_lang = "FR"
    else:
        input_lang = "EN"

    results = []

    direct_matches_en = find_lines_in_index(en_fr_index, cleaned_word)
    for line in direct_matches_en:
        if not translation_has_letters(line):
            continue
        results.append(f"EN -> FR: {line}")

    direct_matches_fr = find_lines_in_index(fr_en_index, cleaned_word)
    for line in direct_matches_fr:
        if not translation_has_letters(line):
            continue
        results.append(f"FR -> EN: {line}")

    for lemma in lemmatize_word_english(cleaned_word):
        lemma_matches_en = find_lines_in_index(en_fr_index, lemma)
        for line in lemma_matches_en:
            if not translation_has_letters(line):
                continue
            res_line = f"EN -> FR: {line}"
            if res_line not in results:
                results.append(res_line)

    for lemma in lemmatize_word_french(cleaned_word):
        lemma_matches_fr = find_lines_in_index(fr_en_index, lemma)
        for line in lemma_matches_fr:
            if not translation_has_letters(line):
                continue
            res_line = f"FR -> EN: {line}"
            if res_line not in results:
                results.append(res_line)

    if not results:
        reverse_results = []
        rev_label = "EN -> FR:" if input_lang == "EN" else "FR -> EN:"

        reverse_matches_en = find_lines_after_colon_from_lines(EN_FR_LINES, cleaned_word)
        for line in reverse_matches_en:
            if not translation_has_letters(line):
                continue
            parts = line.split("::", 1)
            if len(parts) == 2:
                swapped = parts[1].strip() + " :: " + parts[0].strip()
            else:
                swapped = line
            reverse_results.append(f"{rev_label} {swapped}")

        reverse_matches_fr = find_lines_after_colon_from_lines(FR_EN_LINES, cleaned_word)
        for line in reverse_matches_fr:
            if not translation_has_letters(line):
                continue
            parts = line.split("::", 1)
            if len(parts) == 2:
                swapped = parts[1].strip() + " :: " + parts[0].strip()
            else:
                swapped = line
            reverse_results.append(f"{rev_label} {swapped}")

        results.extend(reverse_results)

    seen_keys = set()
    deduped_results = []
    for res in results:
        res_clean = remove_phonetics(res)
        res_clean = remove_extra_spaces(res_clean)
        dedup_key = res_clean.strip()
        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            deduped_results.append(res_clean)

    if deduped_results:
        return "\n".join(deduped_results)
    return "Translation not found."

@app.route('/translate', methods=['GET'])
def translate():
    word = request.args.get('word')
    if not word:
        return jsonify({'error': 'No word provided'}), 400
    result = translate_word(word)
    return jsonify({'translation': result}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
