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
import math

from pathlib import Path
from main import tokenize_and_stem, make_ngram_key
from urllib.parse import urlsplit, unquote
from search_extras import load_extras, rerank_with_links

IMPORTANT_TAG_BOOST = 2

def load_index(index_dir: Path) -> dict:
    """
    load token offsets without loading all postings into memory.
    """
    lexicon_path = index_dir / 'lexicon.jsonl'
    index_path = index_dir / 'inverted_index.jsonl'

    offsets = {}

    with lexicon_path.open('r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            offsets[record['token']] = record['offset']

    return {
        'path': index_path,
        'offsets': offsets
    }

def load_url_mapping(mapping_path: Path) -> dict:
    """
    load document ids and their corresponding urls.
    """
    mapping = {}

    with mapping_path.open('r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            mapping[str(record['doc_id'])] = record['url']

    return mapping

def process_query(query: str) -> list[str]:
    """
    Apply the same lowercase alphanumeric tokenization and stemming in main.py to the query.
    """
    return tokenize_and_stem(query)

def get_postings(index: dict, token: str) -> list:
    """
    retrieve one token's postings directly from disk.
    returns (doc_id, tf, important_tf).
    """
    offset = index['offsets'].get(token)

    if offset is None:
        return []

    with index['path'].open('rb') as file:
        file.seek(offset)
        line = file.readline()

    record = json.loads(line)

    if record['token'] != token:
        raise ValueError(f'Incorrect index offset for {token}')

    return [
        (
            int(posting[0]),
            int(posting[1]),
            int(posting[2]),
            posting[3] if len(posting) > 3 else []
        )
        for posting in record['postings']
    ]

def boolean_and_search(tokens: list[str], index: dict) -> list[int]:
    """
    Returns a list of all the docIDs that match the query tokens
    """
    if not tokens:
        return []

    match = None
    for token in tokens:
        posting = get_postings(index, token)
        doc_ids = [entry[0] for entry in posting]

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
    for doc_id in doc_ids[:5]:
        url = url_mapping.get(str(doc_id))

        if url is not None:
            print(f'{i}. {url}')
            i+=1
# ---------------------------------------------------------
# TF - IDF
# ---------------------------------------------------------

def calculate_tfidf(tf, important_tf, df, total_docs):
    """
    calculate weighted tf-idf with saturated term frequencies.
    prevent repetitive documents from dominating the results.
    """
    if tf <= 0 or df <= 0:
        return 0.0

    # cap repeated terms to reduce excessive frequency effects
    regular_tf = min(tf, 20)
    important_tf = min(important_tf, 5)

    weighted_tf = (
        regular_tf +
        IMPORTANT_TAG_BOOST * important_tf
    )

    tf_weight = 1 + math.log10(weighted_tf)

    idf = 1 + math.log10(
        (total_docs + 1) / (df + 1)
    )

    return tf_weight * idf

def calculate_url_boost(query_tokens, url):
    """
    reward urls containing query terms or a query phrase.
    """
    parsed = urlsplit(url)
    path = unquote(parsed.path).lower()

    # tokenize the url path using the same stemmer
    url_tokens = set(
        process_query(path.replace('-', ' ').replace('_', ' '))
    )

    unique_terms = set(query_tokens)

    matches = len(unique_terms & url_tokens)

    # more matching terms in the url means greater relevance
    boost = 2.0 * matches

    # reward paths containing the full query phrase
    phrase = "-".join(query_tokens)

    stemmed_path = "-".join(
        process_query(path.replace('-', ' ').replace('_', ' '))
    )

    if phrase and phrase in stemmed_path:
        boost += 5.0

    return boost

def count_exact_phrases(query_tokens, positions_by_token):
    """
    count occurrences where every query term appears
    consecutively and in the correct order.
    """
    if len(query_tokens) < 2:
        return 0

    first_positions = positions_by_token.get(query_tokens[0], [])

    remaining = [
        set(positions_by_token.get(token, []))
        for token in query_tokens[1:]
    ]

    count = 0

    for start in first_positions:
        if all(
            start + offset in positions
            for offset, positions in enumerate(remaining, start=1)
        ):
            count += 1

    return count

def rank_results(
    query_tokens,
    matching_doc_ids,
    index,
    total_docs,
    url_mapping
):
    """
    combine weighted tf-idf, url matching,
    n-gram matching, and exact phrase matching.
    """
    if not matching_doc_ids:
        return []

    unique_tokens = list(dict.fromkeys(query_tokens))

    scores = {
        doc_id: 0.0
        for doc_id in matching_doc_ids
    }

    document_positions = {
        doc_id: {}
        for doc_id in matching_doc_ids
    }

    # 1. existing weighted tf-idf
    for token in unique_tokens:
        postings = get_postings(index, token)
        df = len(postings)

        for doc_id, tf, important_tf, positions in postings:
            if doc_id not in scores:
                continue

            scores[doc_id] += calculate_tfidf(
                tf,
                important_tf,
                df,
                total_docs
            )

            document_positions[doc_id][token] = positions

    # 2. 2-gram and 3-gram matching
    for n in (2, 3):
        for i in range(len(query_tokens) - n + 1):
            ngram = make_ngram_key(query_tokens[i:i + n])
            postings = get_postings(index, ngram)
            df = len(postings)

            for doc_id, tf, important_tf, positions in postings:
                if doc_id not in scores:
                    continue

                ngram_score = calculate_tfidf(
                    tf,
                    0,
                    df,
                    total_docs
                )

                scores[doc_id] += 1.5 * ngram_score

    # 3. positional exact phrase matching
    for doc_id in scores:
        phrase_count = count_exact_phrases(
            query_tokens,
            document_positions[doc_id]
        )

        if phrase_count > 0:
            scores[doc_id] += (
                5.0 + 2.0 * min(phrase_count, 5)
            )

    # 4. preserve existing url boosting
    for doc_id in scores:
        url = url_mapping.get(str(doc_id), "")

        scores[doc_id] += calculate_url_boost(
            unique_tokens,
            url
        )

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0])
    )

def display_ranked_results(ranked_results: list, url_mapping: dict):
    """
    Display the top 5 documents ranked by combined relevance.
    """

    if not ranked_results:
        print("No matching documents found.")
        return

    print("\nTop 5 Ranked Search Results:\n")

    for rank, (doc_id, score) in enumerate(ranked_results[:5], start=1):
        url = url_mapping.get(str(doc_id), "URL not found")

        print(f"{rank}. {url}")
        print(f"   Relevance Score: {score:.4f}\n")


def execute_query(query, index, url_mapping, total_docs, extras=None):
    """Shared search entry point for terminal and web GUI."""
    tokens = process_query(query)
    if not tokens:
        return 0, []
    matching_ids = boolean_and_search(tokens, index)
    if not matching_ids:
        return 0, []
    ranked = rank_results(tokens, matching_ids, index, total_docs, url_mapping)
    ranked = rerank_with_links(tokens, ranked, extras)
    return len(matching_ids), ranked[:5]


def search_loop(index, url_mapping, total_docs, extras=None):
    """
    accept queries and display the top five ranked matches.
    """
    print("Search engine ready. Type 'exit' to quit.")

    while True:
        try:
            query = input("Search query: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        tokens = process_query(query)

        if not tokens:
            print("Please enter a valid query.")
            continue

        match_count, ranked_results = execute_query(
            query, index, url_mapping, total_docs, extras
        )
        if not match_count:
            print("No matching documents found.")
            continue
        print(f"\nFound {match_count} matching documents.")
        display_ranked_results(ranked_results, url_mapping)

def main():
    if len(sys.argv) > 2:
        raise SystemExit(
            "Usage: python search.py [index_output_directory]"
        )

    output_dir = (
        Path(sys.argv[1])
        if len(sys.argv) == 2
        else Path('index_output')
    )

    # load the lexicon and url mapping
    index = load_index(output_dir)

    url_mapping = load_url_mapping(
        output_dir / 'url_mapping.jsonl'
    )

    # read the original document count from milestone 1
    with (output_dir / 'index_stats.json').open(
        'r', encoding='utf-8'
    ) as file:
        stats = json.load(file)

    total_docs = stats['documents']

    print(f"Loaded {len(index['offsets']):,} tokens")
    print(f"Loaded {len(url_mapping):,} URLs")

    extras = load_extras(output_dir, total_docs)
    if extras:
        print("Loaded PageRank, HITS graph, and anchor-text index")
    else:
        print("Link analysis files not found; using existing ranking")
    search_loop(index, url_mapping, total_docs, extras)

if __name__ == '__main__':
    main()
