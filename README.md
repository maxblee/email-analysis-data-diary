# Analysis of Colorado's Emails

This is a data diary for an analysis I conducted of the language
Sen. Cory Gardner and Gov. John Hickenlooper used in their
emails during their Senate race in Colorado. As my story
shows, both candidates [used partisan language in their emails](https://coloradosun.com/2020/09/04/cory-gardner-john-hickenlooper-campaign-messaging/), even as they (and
especially Cory Gardner) were marketing themselves as bipartisan candidates.

The partisan language is a theme of political emails in general, as
politicians tend to message emails toward supporters.

## Installation and Setup

**Note: These instructions would replicate the analysis I conducted. However, because of privacy and security concerns, I have not published the
underlying data.**

**Additionally, some of the filenames are little
bit different from those mentioned here. I wound up re-running this analysis after receiving updated data but for simplicity, I'm referring to this as if I just ran the analysis once.**

In order to set up this analysis, you will need to have Python installed
and you will need to install the dependencies. You can do this using
the requirements file (e.g. `pip install -r requirements.txt`) or
by using

```sh
poetry install
```

Additionally, you would need to set up `jupyter` with your
virtual environment, using

```sh
ipython kernel install --user --name=email-analysis
```

And finally, you need to install some dependencies for `nltk` —
specifically, "averaged_perceptron_tagger," which finds the parts
of speech for various words, "wordnet," which lemmatizes words
(or reduces them to their root form), and "punkt," which tokenizes sentences.
You can download those by typing this in Python:

```python
import nltk
for nltk_depend in ["averaged_perceptron_tagger", "punkt", "wordnet"]:
    nltk.download(nltk_depend)
```

I conducted my analysis after receiving two MBox exports from a reporter, Sandra Fish, at
the Colorado News Collaborative. Combined, they contained several hundred emails.

In order to parse these as text, I first wrote a program that converted the MBox data
into raw EML files (or email files):

```sh
$ poetry run python form_eml_data.py data/raw/gardner-single --eml --prefix gardner
$ poetry run python form_eml_data.py data/raw/hickenlooper-single --eml --prefix hickenlooper
```

From there, I converted the EML files into a CSV that would be easier to analyze
using `pandas`:

```sh
$ poetry run python form_eml_data.py data/raw/gardner/ data/raw/eml_gardner.csv --raw
$ poetry run python form_eml_data.py data/raw/hickenlooper/ data/raw/eml_hickenlooper.csv --raw
```

Those commands transformed the EML files into a CSV with the following fields:

- ID: int
- slug: str - This is just the name of the files, minus any information about the directory location
    or file extension (e.g. `gardner-45`)
- subject: str - The subject line of the email
- from: email - The email address of the sender of the email (e.g. `info@coryforco.com`)
- date: YYYY-MM-DD HH:MM:SS+00:00 formatted date - The date and time when the email was sent. (More specifically, this refers to the
    "Date" field in the header of an EML file. According to [RFC 4021](https://tools.ietf.org/html/rfc4021#section-2.1.1), this means "the date and time at which the creator of the message
    indicated that the message was complete and ready to enter the mail delivery system.")
- text: str - The raw text of the email
- html: HTML - If there was HTML included in the body of the message, I also stored that, although
    I didn't wind up using it for my analysis.

As I began exploring the data, I noticed that there were some emails that did not
appear to be related to the campaigns (mostly press releases that
got jumbled with the campaign emails when the reporter searched for and exported the files).
In order to deal with this, I used these two commands to sift through the emails, manually
entering "y" or "n" after each email to state whether it was a campaign email:

```sh
$ poetry run python form_eml_data.py data/raw/eml_gardner.csv data/raw/eml_gardner_validity.csv --validity
$ poetry run python form_eml_data.py data/raw/eml_hickenlooper.csv data/raw/eml_hickenlooper_validity.csv --validity
```

This file added two fields to each document. One, `valid_message`, is a boolean field referring to whether
the message was a campaign email. The other, `candidate` was a string field referring to the name of the candidate.

## Data Cleaning

At that point, I began cleaning the data. The first
step in my cleaning involved combining the Hickenlooper
and Gardner CSV files, adding a field clarifying the name of the candidate, and converting the datetimes from UTC
into Colorado time, using `pandas` and `pytz`. Here's the code I used to run that (slightly modified because I updated the analysis and re-ran it):

```python
import pandas as pd
import pytz

hickenlooper = pd.read_csv("data/raw/eml_hickenlooper_validity.csv", index_col=0)
gardner = pd.read_csv("data/raw/eml_gardner_validity.csv", index_col=0)
combined_candidates = hickenlooper.append(gardner, ignore_index=True)
combined_candidates["date"] = (
    pd.to_datetime(combined_candidates.date, 
    # all of the dates were in YYYY-MM-DD HH:MM:SS+00:00 format, which this has no problem parsing
    infer_datetime_format=True, utc=True)
    .dt.tz_convert(pytz.timezone("America/Denver"))
)
```

From there, I removed any emails that were not sent from the two campaigns,
any emails I realized were not campaign emails during my manual search,
and any emails that had null values. (I wound up running into three
emails that had null values (all in the text field).):

```python
email_domain = lambda addr: addr.split("@")[1]
campaign_domains = {
    "hickenlooper.com",
    "coryforco.com",
    "hello.hickenlooper.com",
    "corygardner.com",
    "corygardnerforsenate.com"
}
combined_candidates = (
    combined_candidates[combined_candidates.valid_message]
    .dropna()
)
combined_candidates = combined_candidates[
    (combined_candidates["from"].apply(email_domain).isin(campaign_domains)) &
    (~combined_candidates["from"].str.startswith("press"))
]
```

I then removed duplicates from the data (all of these came about when I updated the analysis with new data):

```python
# this is needed because the slug is unique
duplicate_fields = ["subject", "from", "date", "text", "html", "candidate"]
combined_candidates = combined_candidates.drop_duplicates(subset=duplicate_fields)
```

Then, I set the date ranges so that all of the emails were from the same time period.

```python
comparison_start = combined_candidates.groupby("candidate").date.min().max()
combined_candidates = combined_candidates[combined_candidates.date >= comparison_start]
```

The last cleaning I did involved removing boilerplate language from the email text. 
(These were email signatures, privacy notices and things of the like that would distract any 
algorithm that's trying to find the words that make one set of documents unique from another.) 

I used the `preprocess` function in `email_analysis/tokenize` to do this. The code likely looks a bit confusing, especially to people without any natural language processing background. But it essentially converts
each document into a list of words, doing a little bit to consolidate words into their root forms (e.g. "dogs" to "dog") and to convert things like numbers and addresses into single tags that computers can more easily make sense of. Then, it creates a counter of all of the 10-word-long phrases in each candidates' emails. And finally, it removes phrases that appeared in more than a certain percentage of the documents.

I borrowed this idea from an *explicit plagiarism detection technique*, as is described in [this review of plagiarism detection techniques](https://arxiv.org/pdf/1801.06323.pdf). 

I settled on thresholds of 8% for Gardner and 7% for Hickenlooper (meaning that I removed any 10-word phrases that appeared in more than 8 and 7 percent of the documents, respectively) after manually sifting through CSVs showing each candidate's 10-word phrases and the proportion of documents they were in. I used `difflib.ndiff` to compare the words that appeared in the cleaned text with the words that appeared in the original text. That allowed me to validate that I only removed boilerplate text. The numbers I reported also exclusively refered to document frequencies and raw word counts. So none of the numbers I reported were affected when I removed this boilerplate text.

Here's the code I used to do this:

```python
from email_analysis import tokenize

hickenlooper = combined_candidates[combined_candidates.candidate == "hickenlooper"].copy()
gardner = combined_candidates[combined_candidates.candidate == "gardner"].copy()
hickenlooper["clean_text"] = tokenize.preprocess(hickenlooper, threshold=0.07, n=10)
gardner["clean_text"] = tokenize.preprocess(gardner, threshold=0.08, n=10)
combined_candidates = gardner.append(hickenlooper)
combined_candidates.to_csv("data/clean/fundraising_emails.csv")
```

At this point, my data was clean enough to analyze. I've included my analysis code in
[`analysis/Email-Fundraising-Analysis.ipynb`](analysis/Email-Fundraising-Analysis.ipynb).
