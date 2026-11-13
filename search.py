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
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

IMPORTANT_TAG_BOOST = 2

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
# ---------------------------------------------------------
# TF - IDF
# ---------------------------------------------------------

def get_document_frequency(index: dict, token: str) -> int:
    """
    Calculates number of documents containing the token
    """
    return len(index.get(token, []))

def calculate_idf(token: str, index: dict, total_docs: int) -> float:
    """
    Calculate inverse document frequency.

    Uses the entire document collection.
    """
    df = get_document_frequency(index, token)

    if df == 0:
        return 0.0

    return math.log10(total_docs / df)

def get_important_frequency(important_index: dict,
                            token: str,
                            doc_id: int
                            ) -> int:
    """
    Return the frequency of a token in important HTML tags
    for a specific document.
    """

    for posting_doc_id, frequency in get_postings(
        important_index, token
    ):

        if posting_doc_id == doc_id:
            return frequency

    return 0

def calculate_weighted_tf(token: str,
                          doc_id: int,
                          index: dict,
                          important_index: dict
                         ) -> float:
    """
    Calculate weighted term frequency.

    Body frequency + additional important-tag frequency.
    """

    # Find body token's term frequency (tf)
    body_frequency = 0

    for posting_doc_id, frequency in get_postings(index, token):

        if posting_doc_id == doc_id:
            body_frequency = frequency
            break

    if body_frequency == 0:
        return 0.0

    # Find important-tag tokens' tf
    important_frequency = get_important_frequency(important_index,
                                                  token,
                                                  doc_id)

    weighted_frequency = (
        body_frequency
        + IMPORTANT_TAG_BOOST * important_frequency
    )

    return 1 + math.log10(weighted_frequency)

def calculate_tf_idf(token: str, 
                    doc_id: int, 
                    index: dict,
                    important_index: dict, 
                    total_docs: int) -> float:

    """
    Calculate weighted TF-IDF for a document term.
    """

    tf = calculate_weighted_tf(token, doc_id, index, important_index)

    idf = calculate_idf(token, index, total_docs)

    return tf * idf

def rank_results(query_tokens: list,
                 matching_doc_ids: list,
                 index: dict,
                 important_index: dict,
                 total_docs: int) -> list:
    """
    Ranks matching documents using TF-IDF and cosine similarity.
    Scores range from -1 (least relevant) to 1 (most relevant).
    Returns ranked results based on tf-idf scoring.
    """

    if not matching_doc_ids or not query_tokens:
        return []

    # Use unique query terms as vector dimensions.
    # Removes duplicate terms from query_tokens and preserves order.
    query_terms = list(dict.fromkeys(query_tokens))

    # Calculate query term frequencies.
    query_frequencies = Counter(query_tokens)

    # Build query TF-IDF vector.
    query_vector = []

    for token in query_terms:
        tf = 1 + math.log10(query_frequencies[token])
        idf = calculate_idf(token, index, total_docs)

        query_vector.append(tf * idf)

    # Build document TF-IDF vectors.
    document_vectors = []

    for doc_id in matching_doc_ids:
        document_vector = []

        for token in query_terms:
            tfidf = calculate_tf_idf(
                token,
                doc_id,
                index,
                important_index,
                total_docs
            )

            document_vector.append(tfidf)

        document_vectors.append(document_vector)

    # Calculate cosine similarity.
    similarities = cosine_similarity(
        np.array([query_vector]),
        np.array(document_vectors)
    )[0]

    # Pair document IDs with similarity scores.
    ranked_results = list(zip(matching_doc_ids, similarities))

    # Sort by highest similarity.
    ranked_results.sort(
        key=lambda result: result[1],
        reverse=True
    )

    return ranked_results

def display_ranked_results(ranked_results: list, url_mapping: dict):
    """
    Display the top 5 documents ranked by cosine similarity.
    """

    if not ranked_results:
        print("No matching documents found.")
        return

    print("\nTop 5 Ranked Search Results:\n")

    for rank, (doc_id, score) in enumerate(ranked_results[:5], start=1):
        url = url_mapping.get(str(doc_id), "URL not found")

        print(f"{rank}. {url}")
        print(f"   Cosine Similarity: {score:.4f}\n")

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

        ranked_results = rank_results(query_tokens, 
                                    matching_doc_ids,
                                    index, 
                                    important_index,
                                    total_docs)
    
        # To do:
        # Display matching URLs using url_mapping
        # JSON object keys are strings, so use url_mapping.get(str(doc_id)).
        display_ranked_results(rank_results, url_mapping)
        # display_results(matching_document_ids, url_mapping)

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
