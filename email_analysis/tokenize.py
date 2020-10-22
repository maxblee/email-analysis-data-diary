import collections
import re
import unicodedata

import commonregex
import nltk
from nltk.corpus import wordnet
import nltk.tokenize
from nltk import stem
import pandas as pd

def add_subtokens(raw_text):
    """This takes a piece of text and converts portions matching various regular expressions.

    This is designed to replace things like numbers that natural
    language processing techniques (at least the ones I'm using)
    can't understand with tokens.
    """
    replace_text = re.sub(r"\s+", " ", raw_text)
    tok_replace = lambda token: f"_<{token}>_"
    remove_patterns = {
        "msg_string": re.compile(r"\-+\s*Forwarded\s+message\s+\-+"),
        "digit": re.compile(r"[0-9]+")
    }
    remove_patterns.update(commonregex.regexes) 
    # need to customize order so that email domains don't get tagged as links, etc.
    pattern_order = [ 
        "links",
        "emails",
        "street_addresses", 
        "phones",
        "dates",
        "times",
        "prices",
        "btc_addresses",
        "credit_cards",
        "hex_colors",
        "ipv6s",
        "ips",
        "msg_string",
        "digit"
    ]
    for token in pattern_order:
        replace_text = remove_patterns[token].sub(tok_replace(token), replace_text)
    return replace_text

def nltk_pos_to_wordnet(pos_tag):
    """Transforms an NLTK pos tag to a WordNet tag.
    
    This helps reduce words that have the same root form
    (e.g. dog and dogs) into the same word.
    """
    # wordnet descriptions from https://wordnet.princeton.edu/documentation/wndb5wn
    # nltk (penn treebank) descriptions from nltk.help.upenn_tagset()
    if pos_tag.startswith("VB"):
        return wordnet.VERB
    if re.match(r"W?RB[A-Z]?", pos_tag):
        return wordnet.ADV
    if pos_tag.startswith("JJ"):
        return wordnet.ADJ
    if re.match(r"NN|PR|WP\$?", pos_tag):
        return wordnet.NOUN
    return None

def tokenize_email(merged_text: str, return_positions: bool=False):
    """Takes a string representing the content of an email
        and yields tokens for every word in that email.

    Args:
        merged_text: A string combination of the subject line and body
            of an email
        return_positions: If you want to *just* return the tokens, set
            to False. If you also want the positions of those tokens set to true.
    """
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    lemmatizer = stem.WordNetLemmatizer()
    for sent_start, sent_end in sent_tokenizer.span_tokenize(merged_text):
        sent = merged_text[sent_start:sent_end]
        sent_words = list(tokenize_sentence(
            sent,
            return_positions=True,
            sentence_start=sent_start
        ))
        just_words = [w[0] for w in sent_words]
        for (word, pos), (_tok, start, end) in zip(nltk.pos_tag(just_words), sent_words):
            wn_pos = nltk_pos_to_wordnet(pos)
            if re.match(r"^\w+$", word) and wn_pos is not None:
                lem_word = lemmatizer.lemmatize(word, wn_pos)
            else:
                lem_word = word
            yield (lem_word, start, end) if return_positions else lem_word

def tokenize_sentence(sentence: str, return_positions=False, sentence_start=None):
    """Takes a sentence and tokenizes it, yielding individual words/tokens.

    Args:
        sentence: A sentence (likely tokenized by tokenize_email)
        return_positions: Optionally, return the positions of each token. 
            (Makes it easier to remove the tokens or extract the original text).
        sentence_start: The start position of the sentence. Required to get
            the text-level positions of each token.
    """
    if return_positions and sentence_start is None:
        raise ValueError("If you want to return the positions of the sentence, include the position of the start of the sentence.")
    sent_has_toks = False
    tokenizer = re.compile(r"_<\\?\w+>_|\w+|[\'\.,!?;—–%\/`~]")
    for count, pattern in enumerate(tokenizer.finditer(sentence)):
        sent_has_toks = True
        token = pattern.group(0).lower()
        if return_positions:
            start_pos = sentence_start + pattern.start()
            end_pos = sentence_start + pattern.end()
        if count == 0:
            # ties start sentence token to the position of the first token
            # so decomposing [('<s>', 19, 19), ('india', 19, 24)] gets str[19:24]
            # e.g. India
            yield ("<s>", start_pos, start_pos) if return_positions else "<s>"
        yield (token, start_pos, end_pos) if return_positions else token
    if sent_has_toks:
        # same idea as start sentence token; ties to end of sentence
        yield ("</s>", end_pos, end_pos) if return_positions else "</s>"

def get_ngrams(sentence_tokens, n=1):
    """Returns a generator of n-gram tuples given a sentence or document.
    
    This allows you to take a list of words and generate a list of
    e.g. 3-word phrases that appeared in that list for any arbitrary
    positive integer `n`.
    """
    n_forward = [sentence_tokens[gram:] for gram in range(n)]
    return zip(*n_forward)

def get_ngram_positions(sentence_tokens, n=1):
    """Returns a dictionary matching sentence tokens to all of their positions.
    
    Args:
        sentence_tokens: A list of tuples in the form (token, start_position, end_position)
    """
    ngrams = collections.defaultdict(list)
    for tok_idx in range(len(sentence_tokens)):
        if tok_idx + n < len(sentence_tokens):
            ngram_end = sentence_tokens[tok_idx + (n - 1)][2]
            ngram_start = sentence_tokens[tok_idx][1]
            tokens = [sentence_tokens[i][0] for i in range(tok_idx, tok_idx + n)]
            ngrams[tuple(tokens)].append((ngram_start, ngram_end))
    return ngrams
        
def remove_boilerplate(tagged_texts, n=10, threshold=0.2):
    """Take an iterator of text and yield those texts without the boilerplate."""
    num_docs = len(tagged_texts)
    freq_vals = {}
    ngram_dicts = []
    unique_ngrams = collections.Counter()
    for text in tagged_texts:
        tokens = list(tokenize_email(text, return_positions=True))
        ngram_positions = get_ngram_positions(tokens, n=n)
        unique_ngrams.update(ngram_positions.keys())
        ngram_dicts.append(ngram_positions)
    skip_tokens = set()
    for tok, count in unique_ngrams.items():
        if count not in freq_vals:
            freq_vals[count] = count / num_docs
        if freq_vals[count] >= threshold:
            skip_tokens.add(tok)
    for ngram_dict, raw_text in zip(ngram_dicts, tagged_texts):
        removal_toks = get_removal_tokens(ngram_dict, skip_tokens)
        yield remove_tokens(raw_text, removal_toks)

def get_removal_tokens(position_dict, skip_tokens):
    """Returns an array of (start, end) tuples representing the parts of
        a text document containing boilerplate text.
    
    Args:
        position_dict: A defaultdict containing all of the unique
            n-grams and their indexes.
        skip_tokens: a set of unique tokens that you want to remove.
    """
    skip_frames = []
    # merge all of the indexes into a single sorted list of (start, end) tuples
    # where the tuples represent parts of the text document we want to get rid of
    skip_vals = sorted(sum(
        (vals for key, vals in position_dict.items() if key in skip_tokens),
        []
    ))
    cur_start = None
    cur_end = None
    for tok_idx in range(len(skip_vals)):
        start, end = skip_vals[tok_idx]
        if cur_start is None:
            cur_start = start
        # we're checking the previous end to the current start token
        # to see if the two tuples are overlapping
        prev_end = skip_vals[tok_idx - 1][1] if tok_idx > 0 else start
        if start <= prev_end:
            # if the tuples overlap, we extend the start + end tuple
            # to cover both
            cur_end = end
        else:
            # o/w we determine that this is the final position of the boilerplate
            skip_frames.append((cur_start, cur_end))
            cur_start = start
            cur_end = end
    # need an extra append to handle (very frequent) cases where the else part of this
    # never happens
    if cur_start is not None and cur_end is not None:
        if len(skip_frames) == 0 or skip_frames[-1] != (cur_start, cur_start):
            skip_frames.append((cur_start, cur_end))
    return skip_frames

def remove_tokens(text, boilerplate_tokens):
    """Takes a string and removes boilerplate based on a list of boilerplate tokens.
    
    (The boilerplate tokens are derived from `get_removal_tokens`.)
    """
    stripped_string = ""
    token_start = 0
    token_end = len(text)
    for start_pos, end_pos in zip(*zip(*boilerplate_tokens)):
        stripped_string += text[token_start:start_pos]
        token_start = end_pos
    stripped_string += text[token_start:token_end]
    return stripped_string

def preprocess(email_data: pd.DataFrame, n=10, threshold=0.2):
    """Takes a pandas DataFrame of emails and extracts the cleaned text from them."""
    raw_text = email_data.subject + ".\n" + email_data.text
    tagged_text = raw_text.apply(add_subtokens)
    return pd.Series(
        remove_boilerplate(tagged_text, n=n, threshold=threshold),
        index=email_data.index
    )