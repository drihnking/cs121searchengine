import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from collections import defaultdict

STEMMER = PorterStemmer()
TOKEN_PATTERN = re.compile(r'[a-zA-Z0-9]+')
IMPORTANT_TAGS = ('title', 'h1', 'h2', 'h3', 'b', 'strong')

def find_files(p: Path):
    """
    Finds all files in directory and subdirectories recursively
    Maps each document_id to its parsed data
    documents = { document_id : }
    """
    # Maps each numeric document_id to its parsed document data
    documents = []

    def visit(directory: Path):
        # Turns folder into a list of Path files
        for item in directory.iterdir():
            if item.is_dir():
                visit(item)

            # Only JSON files to index
            elif item.is_file() and item.suffix.lower() == '.json':
                try:
                    # Assign one ID to each successfully parsed document
                    documents.append(read_file(item))
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    # Skip invalid files
                    print(f'Skipping {item}: {exc}', file = sys.stderr)

    visit(p)
    return documents

def read_file(p: Path):
    """
    Opens and reads file, then parses HTML content.
    """
    with p.open('r', encoding = 'utf-8-sig') as file:
        return parse_file(file)

def parse_file(file):
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
        #'url_id': document_id, # To do: replace with stable numeric ID if necessary
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


def build_index(documents):
    '''
    Maps each token to the set of document_ids containing that token
    inverted_index = { token : set of document_ids }
    url_mapping = { document_id : url}
    '''
    # Maps each token to the set of document_ids containing that token
    inverted_index = defaultdict(set) # dictionary with key=token, value = docid
    
    # Maps each document_id back to its url for search results
    url_mapping = {}

    # Parse each text document in "documents" list
    for document_id, document in enumerate(documents):
        url_mapping[document_id] = document['url']
        # Use set to prevent duplicate tokens
        unique_tokens = set(document['tokens'])
        for token in unique_tokens:
            inverted_index[token].add(document_id)
    
    return inverted_index, url_mapping

# Main function to run when gathering data.
if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python main.py /path/to/developer.zip')
    corpus_path = Path(sys.argv[1])
    if not corpus_path.is_dir():
        raise SystemExit(f'Not a directory: {corpus_path}')

    # Read and parse all files
    documents = find_files(corpus_path)

    # Build the in-memory indexes
    inverted_index, url_mapping = build_index(documents)

    print(inverted_index)


