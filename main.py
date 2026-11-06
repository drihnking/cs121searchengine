import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

STEMMER = PorterStemmer()
TOKEN_PATTERN = re.compile(r'[a-zA-Z0-9]+')
IMPORTANT_TAGS = ('title', 'h1', 'h2', 'h3', 'b', 'strong')

def find_files(p: Path):
    """
    Finds all files in directory and subdirectories recursively
    """
    # Turns folder into a list of Path files
    for item in p.iterdir():
        if item.is_dir():
            find_files(item)

        elif item.is_file() and item.suffix.lower() == '.json':
            try:
                read_file(item)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f'Skipping {item}: {exc}', file = sys.stderr)

def read_file(p: Path):
    """
    Opens and reads file, then parses HTML content.
    """
    with p.open('r', encoding = 'utf-8-sig') as file:
        return parse_file(file, document_id = str(p))

def parse_file(file, document_id = None):
    """
    Extract page data and deal with missing or malformed HTML
    """
    page = json.load(file)
    if not isinstance(page, dict):
        raise ValueError('Expected a JSON object containing page fields')

    url = page.get('url', '')
    html = page.get('content') or ''
    if not isinstance(html, str):
        raise ValueError('Page content must be a string')

    # If html5lib is not installed, fall back to built-in html.parser
    try:
        import html5lib
        parser = 'html5lib'
    except ImportError:
        parser = 'html.parser'
    soup = BeautifulSoup(html, parser)

    # Scripts and styles are not visible page text; should not be indexed
    for tag in soup(['script', 'style', 'noscript', 'template']):
        tag.decompose()

    text = soup.get_text(separator = ' ', strip = True)
    tokens = tokenize_and_stem(text)

    # Keep important tokens separately
    important_tokens = []
    for tag in soup.find_all(IMPORTANT_TAGS):
        # Nested important tags may overlap since this is just preliminary metadata
        important_tokens.extend(tokenize_and_stem(tag.get_text(' ', strip = True)))

    # ------------------ To Do Here: ------------------------
    # Add tokens to an inverted index
    # Save index to disk and calculate the report statistics
    # -------------------------------------------------------

    return {
        'document_id': document_id, # To do: replace with stable numeric ID if necessary
        'url': url.split('#', 1)[0] if isinstance(url, str) else '',
        'tokens': tokens,
        'important_tokens': important_tokens
    }

def tokenize_and_stem(text: str) -> list[str]:
    """

    Lowercase, extract sequences, and stem every token, including stop words.

    """
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [STEMMER.stem(token) for token in tokens]

# Dictionary to trace back document url given docid: key=docid, val=page_url
PAGES_LIST = {} 

def pages_index(pages):
    docid = 0
    for page_url in pages:
        docid += 1
        PAGES_LIST[page_url] = docid

def build_index(pages):
    '''
    Indexes pages containing each token t
    '''
    inverted_list = {} # dictionary with key=token, value = docid
    
    # Parse each text document in "documents" list
    for page_url in pages:
        # Parse the document into tokens
        tokens = parse(doc)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            inverted_list[token] = inverted_list.get(token, []).append(docid)
    
    return inverted_list
    
# Main function to run when gathering data.
if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python main.py /path/to/developer.zip')
    corpus_path = Path(sys.argv[1])
    if not corpus_path.is_dir():
        raise SystemExit(f'Not a directory: {corpus_path}')
    find_files(corpus_path)

