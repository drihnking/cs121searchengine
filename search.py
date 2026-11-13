"""
Loads the JSON index from main.py, normalizes queries with the indexer's own tokenizer,
and shows individual term postings. Includes boolean AND retrieval, ranking, and displaying
matching URLs are also implemented here.

To run this file:
- Keep it in the same folder as main.py
- python search.py or python search.py index_output
"""

import json
import sys
from pathlib import Path
from main import tokenize_and_stem

def load_index(index_path: Path) -> dict:
    """ Read token [(document_id, term_frequency), ...] from a JSON file. """
    with index_path.open('r', encoding = 'utf-8') as file:
        index = json.load(file)

    if not isinstance(index, dict):
        raise ValueError('Inverted index has to be a JSON object')
    return index

def load_url_mapping(mapping_path: Path) -> dict:
    """ Read document ID, return URL from JSON """
    with mapping_path.open('r', encoding = 'utf-8') as file:
        mapping = json.load(file)
    if not isinstance(mapping, dict):
        raise ValueError('URL mapping must be a JSON object')
    return mapping

def process_query(query: str) -> list[str]:
    """
    Apply the same lowercase alphanumeric tokenization and stemming in main.py to the query.
    """
    return tokenize_and_stem(query)

def get_postings(index: dict, token: str) -> list[tuple[int, int]]:
    """ Return (document_id, term_freq) tuples for a token, or [] if doesn't exist"""
    postings = index.get(token, [])
    return [(int(document_id), int(frequency)) for document_id, frequency in postings]

def boolean_and_search(tokens: list[str], index: dict) -> list[int]:
    """
    Returns a list of all the docIDs that match the query tokens
    """
    if not tokens:
        return []

    match = None
    for token in tokens:
        posting = get_postings(index, token)
        doc_ids = [doc_id for doc_id, frequency in posting]

        if match is None:
            match = doc_ids
        else:
            match = _intersect(match, doc_ids)

        # If match does not contain any doc_ids that fufill the query, immediately return
        if not match:
            return []
        
    return match

def _intersect(p1: list[int], p2: list[int]) -> list[int]:
    """
    Compares two postings and returns a list of their matching docIDs
    """
    answer = []

    i, k = 0, 0
    while i<len(p1) and k<len(p2):
        if p1[i]==p2[k]:
            answer.append(p1[i])
            i+=1
            k+=1
        elif p1[i]<p2[k]:
            i+=1
        else:
            k+=1

    return answer

def display_results(doc_ids: list[int], url_mapping: dict):
    """
    Takes a list of docIds and prints its respectively mapped URL
    """
    if not doc_ids:
        print('No results.')
        return

    i=1
    for doc_id in doc_ids:
        url = url_mapping.get(str(doc_id))

        if url is not None:
            print(f'{i}. {url}')
            i+=1

def search_loop(index: dict, url_mapping: dict) -> None:
    """ Accepts console queries and shows token statistics """
    print("Search component is ready. Type a query or 'exit' to quit.")

    while True:
        try:
            query = input('Search query: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye!')
            break

        if query.lower() in ('exit', 'quit'):
            print('Goodbye!')
            break

        tokens = process_query(query)
        if not tokens:
            print('Please enter a query with at least one letter or digit.')
            continue

        # To do:
        # Implement boolean_and_search(tokens,index)
        # Use get_postings(index, token) to get each term's postings
        # If any term is missing, return no matches, otherwise intersect
        # the document IDs so results contain all query terms
        matching_document_ids = boolean_and_search(tokens, index)
    
        # To do:
        # Display matching URLs using url_mapping
        # JSON object keys are strings, so use url_mapping.get(str(doc_id)).
        display_results(matching_document_ids, url_mapping)

def main() -> None:
    if len(sys.argv) > 2:
        raise SystemExit('Usage: python search.py [index_output_directory]')

    output_directory = Path(sys.argv[1]) if len(sys.argv) == 2 else Path('index_output')
    index_path = output_directory / 'inverted_index.json'
    mapping_path = output_directory / 'url_mapping.json'

    if not index_path.is_file() or not mapping_path.is_file():
        raise SystemExit(
            'Could not find inverted_index.json and url_mapping.json in '
            f'{output_directory}. First run your indexer, or pass the correct directory.'
        )

    # This initial implementation loads the entire JSON index into memory.
    # To do: read individual postings from disk instead.
    print('Loading index into memory...', flush = True)
    index = load_index(index_path)
    url_mapping = load_url_mapping(mapping_path)
    print(f'Loaded {len(index):,} indexed terms and {len(url_mapping):,} URL mappings.')

    search_loop(index, url_mapping)


if __name__ == '__main__':
    main()
