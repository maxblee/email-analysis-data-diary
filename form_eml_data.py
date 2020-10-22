import collections
import glob
import mailbox
import os
import pathlib
import re
from typing import Any, Callable, List, Dict
import unicodedata

import eml_parser
import pandas as pd


DATA_DIR = pathlib.Path("./data/")
RAW_DATA = DATA_DIR / "raw"

# refers to the different parts of a MIME email body
# see https://github.com/GOVCERT-LU/eml_parser/blob/master/eml_parser/eml_parser.py#L708
MimePart = collections.namedtuple("MimePart", "encoding body_string headers")

def eml_data(eml_path: str) -> Dict[str, str]:
    """Extracts information about an email.
    
    Takes an EML file and returns a dictionary containing information about the
    metadata, text, and HTML contents found in the headers and body of
    that email.

    **Note:** In my experience, the date returns on this are *typically* accurate.
    However, there are some edge cases that fail using hypothesis, possibly because of
    downstream problems involving `eml_parser`.

    Args:
        eml_path: The file path of the HTML file (e.g. example.eml )
    
    Returns:
        A dictionary of metadata. All of the return values in the metadata are
        strings. Null values are represented as empty strings.
        For example, if your filename is ~/Desktop/example.eml,
        this might return:
        
        {
            "slug": "example.eml",
            "subect": "Testing",
            "from": "example@example.com",
            "date": "2020-06-29T09:52:38-07:00",
            "text": "I\'m just sending an example email.",
            "html": "<div>I\'m just sending an <a href=\"https://www.example.com\">example email.</div>"
        }
    """
    # add information about the HTML, text, and other inline attachments in the body
    parser = eml_parser.EmlParser(include_raw_body=True)
    eml_info = parser.decode_email(eml_path)
    header = eml_info["header"]
    body = eml_info["body"]
    return {
        "slug": get_slug(eml_path),
        "subject": header.get("subject", ""),
        "from": header.get("from", ""),
        # returns datetime in ISO format
        "date": str(header.get("date", "")),
        "text": __get_body_content("text/plain", body),
        "html": __get_body_content("text/html", body),
    }


def __get_body_content(content_type: str, body: dict) -> str:
    """Get the content of a particular content type in an email.

    This is a helper function for get_eml_data and should not be used
    outside of that context.
    """
    content = [
        attachment["content"]
        for attachment in body
        if attachment.get("content_type", "") == content_type
    ]
    return "" if len(content) != 1 else content[0]


def get_slug(filepath: str) -> str:
    """Gets a slug for a file.
    
    Replaces whitespace from a filepath and removes directory
    information and extension information from the path.

    **Note:** This is not particularly robust, as it does not
    handle special HTML characters (e.g. %20 for whitespace)
    and does not remove characters other than whitespace.

    Args:
        filepath: The path to a file you want to get a slug for.

    Returns:
        A string of that file stripped of its extension and directory
        prefix, and with whitespace replaced by hyphens.
    """
    _, rel_path = os.path.split(filepath)
    file_name, _ = os.path.splitext(rel_path)
    sluggify = lambda x: re.sub(r"\s+", "-", unicodedata.normalize("NFKC", x))
    return sluggify(file_name)

def save_raw_eml(
    mbox_file,
    output_directory,
    prefix="hickenlooper",
    verbose=False
):
    """Takes an mbox file and outputs all of the individual EML files
    into a directory.

    Args:
        mbox_file: The path to your mbox file
        output_directory: The path to the directory where you want those files saved
        prefix: The prefix you want added to any slugs
    """
    # needed so ~/.. becomes an absolute path (for some reason is_dir doesn't work o/w)
    output_dir = os.path.expanduser(output_directory)
    if not os.path.isdir(output_dir):
        raise TypeError("You must enter a valid directory to output these files to")
    dir_and_prefix = os.path.join(output_dir, prefix)
    dir_files = glob.glob(os.path.join(output_dir, f"{prefix}-*.eml"))
    eml_pattern = re.compile(r"^{}\-([0-9]+)\.eml$".format(dir_and_prefix))
    # set -1 as default b/c sets iterable to 0
    last_msg_id = max([
        int(eml_pattern.sub(r"\1", fname)) for fname in dir_files
        if eml_pattern.match(fname)
    ], default=-1)
    messages = mailbox.mbox(mbox_file)
    for msg_id, msg in messages.items():
        if verbose:
            print(f"Parsing message {msg_id}")
        slug = f"{prefix}-{msg_id + last_msg_id}"
        try:
            with open(os.path.join(output_dir, f"{slug}.eml"), "w") as f:
                f.write(str(msg))
        except UnicodeEncodeError:
            import logging
            logging.warning(f"Could not parse message {slug}")

def hick_and_gardner_emails(input_dir):
    """Takes an input directory and yields all of the email messages from said directory
    """
    input_dir = os.path.expanduser(input_dir)
    if not os.path.isdir(input_dir):
        raise TypeError("You must enter a valid directory where these EML files are located")
    all_eml_files = glob.glob(os.path.join(input_dir, "*.eml"))
    for eml_file in all_eml_files:
        yield eml_data(eml_file)

def write_raw_csv(
    input_dir,
    output_file,
    reference_csv=None
    ):
    """Writes a raw CSV of all of the metadata (including body text and HTML)
    from emails in a directory.

    Args:
        input_dir: the path to the directory where the EML files are.
        output_file: the filepath for the less-raw file
        reference_csv: If it isn't None, limits output to just slugs that
            aren't in the reference csv
    """
    if reference_csv is not None:
        reference_data = pd.read_csv(reference_csv, index_col=0)
    else:
        # set up an empty DataFrame with a slug column (so x in reference_df.slug.unique() is always False)
        reference_data = pd.DataFrame(columns=["slug"])
    processed_slugs = reference_data.slug.unique()
    all_emails = pd.DataFrame(hick_and_gardner_emails(input_dir))
    new_emails = all_emails[~all_emails.slug.apply(lambda x: x in processed_slugs)]
    new_emails.to_csv(output_file)

def check_message_validity(message_text):
    update_fields = {"candidate": message_text["slug"].split("-")[0]}
    print(f"{message_text['text']}\n")
    response_is_valid = False
    while not response_is_valid:
        response = input("""
            Is this message about the 2020 Senate race between 
            Cory Gardner and John Hickenlooper?\n
            Type Y/y for yes, N/n for no.\n
        """).strip().lower()
        if response == 'y':
            response_is_valid = True
            update_fields["valid_message"] = True
        elif response == 'n':
            response_is_valid = True
            update_fields["valid_message"] = False
        else:
            print("Sorry, you didn't enter a valid response. Please enter Y/y or N/n.")
    update_fields.update(message_text)
    return update_fields
        

def check_messages(
    infile, 
    outfile
    ):
    """Manually check that every message is a fundraising message
    from one of the candidates.
    """
    msg_df = pd.read_csv(infile, index_col=0)
    msgs = (check_message_validity(msg_df.loc[msg]) for msg in msg_df.index)
    clean_df = pd.DataFrame(msgs)
    clean_df.to_csv(outfile)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="""This is a light utility for creating a CSV file
        containing the raw contents of an `mbox` export (or multiple `mbox` exports)
        of emails. 

        The general program is built around two positional arguments, the first for the input
        file, the second for the output file. In addition, you should add a flag representing the
        mode (or functions) you want to run.
        """
    )
    parser.add_argument("input", type=str, help="The path to the input file or directory")
    parser.add_argument("output", type=str, help="The path to the output file or directory")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--eml", action="store_true", help="Save the mbox output into raw EML files.")
    modes.add_argument("--raw", action="store_true", help="Take files in a directory and output their metadata to a CSV")
    modes.add_argument("--validity", action="store_true", help="Write a CSV noting whether each email is a 2020 Senate campaign political fundraising email.")

    # specific to save_raw_eml
    parser.add_argument("--prefix", nargs="?", default="email", help="The prefix for an EML slug")
    parser.add_argument("-v", "--verbose", action="store_true", help="Prints verbose output")

    # specific to write_raw_csv
    parser.add_argument("--reference", nargs="?", default=None, help="The path to a CSV of already-processed files")

    args = parser.parse_args()

    if args.eml:
        save_raw_eml(args.input, args.output, args.prefix, args.verbose)
    elif args.raw:
        write_raw_csv(args.input, args.output, args.reference)
    elif args.validity:
        check_messages(args.input, args.output)
    else:
        raise TypeError("Please add one of the following options: --eml, --raw, --validity")