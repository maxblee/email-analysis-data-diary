"""A module for visually debugging text-based analysis
results.
"""
import bisect

import altair as alt
import pandas as pd

import tokenize

def yield_examples(texts, ngram_tokens, num_examples=None, yield_type=all):
    """Generate texts that match the particular example.
    
    Args:
        texts: a pandas Series of strings (e.g. from co_fundraising.tokenize.preprocess)
        ngram_tokens: a list of tuples representing the n-grams you're searching for
            or a single tuple representing the single n-gram you're searching for
        num_examples: The number of examples you want to generate. If set to None,
            it will return all tokens.
        yield_type: A function that you can run on an iterable to return a boolean value.
            If set to all, it will yield texts only if they include *both* n-grams; if
            set to any, it will yield texts that correspond to any one of the n-grams. Etc.
    Yields:
        Tuples of (the index of the matching text, the whole text, and the position of the ngrams) 
    """
    num_examples = len(texts) if num_examples is None else num_examples
    ngram_tokens = [ngram_tokens] if isinstance(ngram_tokens, tuple) else ngram_tokens
    ngram_size = len(ngram_tokens[0])
    count = 0
    for idx in texts.index:
        text = texts.loc[idx]
        if count >= num_examples:
            break
        tokenized = list(co_fundraising.tokenize.tokenize_email(
            text,
            return_positions=True
        ))
        ngram_pos = co_fundraising.tokenize.get_ngram_positions(
            tokenized,
            n=ngram_size
        )
        if yield_type([tok in ngram_pos for tok in ngram_tokens]):
            count += 1
            yield (idx, text, ngram_pos)

def print_examples(texts, ngram_tokens, num_examples=10, context_window=40, return_vals=False, yield_type=all):
    """Prints up to num_examples examples of a certain n-gram appearing in a pandas Series
    of texts.
    
    Each item it prints comes from a *separate* document. Additionally, the part it prints
    will correspond to the first positional match of the *first token* passed through it.
    
    Args:
        texts: a pandas Series of string texts.
        ngram_tok: a tuple of strings representing a single n-gram
        context_window: the number of characters before and after the n-gram you want to match to.
        return_vals: If true, it will return *all* examples.
    """
    ngram_tokens = [ngram_tokens] if isinstance(ngram_tokens, tuple) else ngram_tokens
    examples = []
    for _idx, text, ngram_pos in yield_examples(texts, ngram_tokens, num_examples, yield_type):
        token_iter = (tok for tok in ngram_tokens if tok in ngram_pos)
        ngram_tok = next(token_iter)
        start, end = ngram_pos[ngram_tok][0]
        start_window = max(0, start - context_window)
        end_window = min(len(text), end + context_window)
        print(text[start_window:end_window])
        examples.append((text, start, end))
    if return_vals:
        return examples

def get_example_df(
    texts, 
    ngram_tokens, 
    reference_df=None, 
    num_examples=None, 
    yield_type=all,
    series_name="text"
):
    """Given a series of texts, a list of n-gram tokens and an optional reference DataFrame.
    
    Args:
        texts: a pandas Series of raw string texts. See yield_examples for details.
        ngram_tokens: a list or string of ngrams. See yield_examples for details.
        reference_df: a reference DataFrame, optionally. If set, this will merge your example series
            to the DataFrame.
        num_examples: the number of matching examples you want to produce. See yield_examples
            for details.
        yield_type: See yield_examples
        series_name: The name you want to give the text column (containing the matching texts).
    Returns:
        A pandas Series if reference_df is not set. Otherwise, a pandas DataFrame.
    """
    matching_texts = []
    indexes = []
    for idx, text, _ngram_dict in yield_examples(texts, ngram_tokens, num_examples, yield_type):
        matching_texts.append(text)
        indexes.append(idx)
    example_series = pd.Series(matching_texts, index=indexes, name=series_name)
    if reference_df is not None:
        return reference_df.join(example_series, rsuffix="_example_series", how="right")
    return example_series

def display_example_date_range(
    texts,
    ngram_tokens,
    reference_df,
    num_examples=None,
    yield_type=all,
    series_name="text",
    variable_category="candidate",
    color_scheme=None
):
    """Displays a tick-style graph showing the date ranges in which matching
    emails were composed.
    
    Args:
        variable_category: The category you want to use as the Y-coordinate
            system for the graphic.
        color_scheme: If set, an instance of alt.Scale for the color scheme of the graphic.
        For the other arguments, see get_example_df
    """
    matching_texts = get_example_df(
        texts, 
        ngram_tokens, 
        reference_df,
        num_examples,
        yield_type,
        series_name
    )
    ngram_tokens = [ngram_tokens] if isinstance(ngram_tokens, tuple) else ngram_tokens
    if not isinstance(matching_texts, pd.DataFrame) or "date" not in matching_texts.columns:
        raise ValueError("You must provide a reference DataFrame with a column called 'date'")
    if color_scheme is None:
        if variable_category == "candidate":
            color_scheme = alt.Color(
                variable_category,
                scale=alt.Scale(
                    domain=["gardner", "hickenlooper"],
                    range=["red", "blue"]
                )
            )
        else:
            color_scheme = f"{variable_category}:O"
    return alt.Chart(matching_texts).mark_tick().encode(
        x="date:T",
        y=f"{variable_category}:O",
        color=color_scheme
    ).properties(title=", ".join([" ".join(tok) for tok in ngram_tokens]))

import bisect

def display_differences(
    difference_func,
    all_doc,
    all_combined,
    gardner_doc,
    gardner_combined,
    hick_doc,
    hick_combined,
    num_display=25,
    verbose=True,
    return_vals=False
):
    """Given a difference func and a whole bunch of DataFrames and Series,
    return a sorted list of the tokens and their corresponding scores.
    
    Args:
        difference_func: The core function you want to return scores for.
            See below for examples.
        all_doc, ..., hick_combined: pandas Series and DataFrames in the
            form returned by get_n_gram_counts
        num_display: The number of results you want to display.
        verbose: Set to true if you want to print the results, False o/w
        return_vals: Set to true if you want to get a return value, False o/w
    """
    differences = []
    for token in all_combined:
        token_score = difference_func(
            token,
            all_doc,
            all_combined,
            gardner_doc,
            gardner_combined,
            hick_doc,
            hick_combined
        )
        # so you can optionally filter out values (e.g. stop words)
        if token_score is not None:
            bisect.insort(differences, (token_score, token))
    high_scores = differences[:num_display]
    low_scores = differences[:-num_display - 1:-1]
    merged = [
        f"{(count + 1):3d}. {' '.join(high[1]):<40} {high[0]:.5f}\t{' '.join(low[1]):<40} {low[0]:.5f}"
        for (count, high), low in zip(enumerate(high_scores), low_scores)
    ]
    header = f"{' ' * 29}Low Scores{' ' * 4}\t{' ' * 30}High Scores"
    if verbose:
        print(header)
        print("\n".join(merged))
    if return_vals:
        return differences
    
def get_frequency_df(
    difference_func,
    all_doc,
    all_combined,
    gardner_doc,
    gardner_combined,
    hick_doc,
    hick_combined,
    num_display=25,
    verbose=False,
    column_names=["Score", "Token", "Token Frequency", "Gardner Frequency", "Hickenlooper Frequency"]
):
    """Returns a DataFrame of all of the tokens, scores, and combined frequencies
    from `display_differences`.
    
    Args:
        column_names: A list of 5 names to give the resulting DataFrame.
        See `display_differences` for other arguments.
    """
    score, token, freq, gardner_freq, hick_freq = column_names
    diff_list = display_differences(
        difference_func,
        all_doc,
        all_combined,
        gardner_doc,
        gardner_combined,
        hick_doc,
        hick_combined,
        verbose=verbose,
        num_display=num_display,
        return_vals=True
    )
    freq_df = pd.DataFrame(diff_list, columns=[score, token])
    freq_df[freq] = freq_df[token].apply(all_combined.freq)
    freq_df[gardner_freq] = freq_df[token].apply(gardner_combined.freq)
    freq_df[hick_freq] = freq_df[token].apply(hick_combined.freq)
    return freq_df

def display_frequency_graph(*args, **kwargs):
    """Displays a scatterplot mapping word scores to their underlying frequencies.
    
    Based on the visualizations from p.377 of http://languagelog.ldc.upenn.edu/myl/Monroe.pdf
    
    Args:
        See `get_frequency_df` for main arguments and keywords. Optionally, you may
        also pass in these kwargs:
            - title: The title of your plot
            - max_size: The maximum number of points you want to plot. This plots out
            the top and bottom n <= max_size / 2 points
    """
    freq_df = get_frequency_df(*args, **kwargs)
    max_size = kwargs.get("max_size")
    if max_size is not None and len(freq_df) > max_size:
        n_top = max_size // 2
        freq_df = freq_df.head(n_top).append(freq_df.tail(n_top))
    score, token, freq, gardner_freq, hick_freq = freq_df.columns
    chart = alt.Chart(freq_df).mark_point().encode(
        x=alt.X(freq, scale=alt.Scale(type="log", base=10)),
        y=score,
        tooltip=[score, token, freq, gardner_freq, hick_freq]
    )
    if "title" in kwargs:
        chart = chart.properties(title=kwargs.get("title"))
    return chart.interactive()