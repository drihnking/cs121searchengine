import json
import re
import sys
import hashlib
import argparse
import heapq
import os 
import tempfile

from pathlib import Path
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from itertools import groupby
from functools import lru_cache

STEMMER = PorterStemmer()
TOKEN_PATTERN = re.compile(r'[a-zA-Z0-9]+')
IMPORTANT_TAGS = ('title', 'h1', 'h2', 'h3', 'b', 'strong')

# implement near & exact duplicate w/ simhash 64-bit
SIMHASH_BITS = 64
SIMHASH_THRESHOLD = 3
SIMHASH_BANDS = 4
MIN_NEAR_DUP_TOKENS = 50

def compute_simhash(tokens):
    """
    calculate a 64-bit simhash using sampled 3-word shingles.
    sampling limits the work required for very large pages.
    """
    if len(tokens) < 3:
        return None

    max_shingles = 2000

    # sample shingles across the entire document
    possible_shingles = len(tokens) - 2
    step = max(
        1,
        (possible_shingles + max_shingles - 1) // max_shingles
    )

    shingles = Counter(
        (tokens[i], tokens[i + 1], tokens[i + 2])
        for i in range(0, possible_shingles, step)
    )

    weights = [0] * SIMHASH_BITS

    for shingle, frequency in shingles.items():
        text = " ".join(shingle).encode("utf-8")

        digest = hashlib.blake2b(
            text, digest_size=8
        ).digest()

        value = int.from_bytes(digest, "big")

        for bit in range(SIMHASH_BITS):
            if value & (1 << bit):
                weights[bit] += frequency
            else:
                weights[bit] -= frequency

    fingerprint = 0

    for bit, weight in enumerate(weights):
        if weight > 0:
            fingerprint |= (1 << bit)

    return fingerprint


def hamming_distance(hash1, hash2):
    """
    count the number of different bits.
    """
    return (hash1 ^ hash2).bit_count()

def find_files(p: Path, stats):
    """
    read documents one at a time and skip duplicate pages.
    yield accepted documents without storing all their tokens.
    """
    seen_urls = set()
    fingerprints = []
    bands = defaultdict(set)

    mask = (1 << 16) - 1

    for root, directories, filenames in os.walk(p):
        # make document processing order reproducible
        directories.sort()

        for filename in sorted(filenames):
            if not filename.lower().endswith('.json'):
                continue

            path = Path(root) / filename
            stats['files_seen'] += 1

            try:
                document = read_file(path)
                url = document['url']

                if not url:
                    stats['invalid_files'] += 1
                    continue

                # reject repeated urls
                if url in seen_urls:
                    stats['duplicate_urls_skipped'] += 1
                    continue

                fingerprint = None
                match_id = None

                tokens = document['tokens']

                if len(tokens) >= MIN_NEAR_DUP_TOKENS:
                    fingerprint = compute_simhash(tokens)

                    candidates = set()

                    # find previously accepted documents
                    # with at least one matching hash band
                    for band_id in range(SIMHASH_BANDS):
                        value = (
                            fingerprint >> (band_id * 16)
                        ) & mask

                        candidates.update(
                            bands.get((band_id, value), set())
                        )

                    # verify candidates using full hamming distance
                    for candidate_id in sorted(candidates):
                        if hamming_distance(
                            fingerprint,
                            fingerprints[candidate_id]
                        ) <= SIMHASH_THRESHOLD:
                            match_id = candidate_id
                            break

                if match_id is not None:
                    stats['near_duplicates_skipped'] += 1
                    seen_urls.add(url)
                    continue

                # assign the next accepted document id
                doc_id = len(fingerprints)

                seen_urls.add(url)
                fingerprints.append(fingerprint)

                # register the fingerprint in the band lookup
                if fingerprint is not None:
                    for band_id in range(SIMHASH_BANDS):
                        value = (
                            fingerprint >> (band_id * 16)
                        ) & mask

                        bands[(band_id, value)].add(doc_id)

                stats['documents'] += 1

                yield document

            except (OSError, ValueError, TypeError, UnicodeError) as exc:
                stats['invalid_files'] += 1
                print(f"Skipping {path}: {exc}", file=sys.stderr)

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

    # use the faster built-in html parser
    soup = BeautifulSoup(html, 'html.parser')

    # Scripts and styles are not visible page text; should not be indexed
    for tag in soup(['script', 'style', 'noscript', 'template']):
        tag.decompose()

    text = soup.get_text(separator = ' ', strip = True)
    tokens = tokenize_and_stem(text)

    # Keep important tokens separately
    important_tokens = []
    for tag in soup.find_all(IMPORTANT_TAGS):

        # avoid counting a child tag twice when its
        # parent is already an important tag
        if tag.find_parent(IMPORTANT_TAGS):
            continue

        important_tokens.extend(tokenize_and_stem(tag.get_text(' ', strip=True)))

    return {
        #'url_id': document_id, # To do: replace with stable numeric ID if necessary
        'url': url.split('#', 1)[0] if isinstance(url, str) else '',
        'tokens': tokens,
        'important_tokens': important_tokens
    }

@lru_cache(maxsize=100000)
def stem_token(token):
    """
    cache stemming results to avoid repeated calculations.
    """
    return STEMMER.stem(token)


def tokenize_and_stem(text: str) -> list[str]:
    """
    lowercase, tokenize and stem all alphanumeric words.
    """
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [STEMMER.stem(token) for token in tokens]


def build_index(documents):
    '''
    Maps each token to the list of document_ids containing that token
    inverted_index = { token : set of document_ids }
    url_mapping = { document_id : url}
    '''
    # Maps each token to the list of document_ids containing that token
    inverted_index = defaultdict(list) # dictionary with key=token, value = docid
    
    # Maps each document_id back to its url for search results
    url_mapping = {}

    # Parse each text document in "documents" list
    for document_id, document in enumerate(documents):
        url_mapping[document_id] = document['url']
        
        # Count frequency of tokens in document
        term_frequencies = {}
        for token in document['tokens']:
            term_frequencies[token] = term_frequencies.get(token, 0) + 1

        for token, term_frequency in term_frequencies.items():
            inverted_index[token].append((document_id, term_frequency))
    
    return inverted_index, url_mapping

# Main function to run when gathering data.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build an optimized inverted index'
    )

    parser.add_argument(
        'corpus',
        help='Directory containing JSON documents'
    )

    parser.add_argument(
        'output',
        nargs='?',
        default='index_output',
        help='Directory for saved index files'
    )

    parser.add_argument(
        '--chunk-docs',
        type=int,
        default=500,
        help='Documents per partial index'
    )

    args = parser.parse_args()

    corpus_path = Path(args.corpus)

    if not corpus_path.is_dir():
        parser.error(f'Invalid corpus directory: {corpus_path}')

    if args.chunk_docs < 1:
        parser.error('--chunk-docs must be positive')

    stats = {
        'files_seen': 0,
        'documents': 0,
        'duplicate_urls_skipped': 0,
        'near_duplicates_skipped': 0,
        'invalid_files': 0
    }

    # find_files now yields one document at a time
    documents = find_files(corpus_path, stats)

    # build, flush, merge, and serialize the index
    stats = build_index(
        documents,
        args.output,
        stats,
        args.chunk_docs
    )

    print('\nMilestone 1 Results')
    print('-------------------')
    print(f"Documents indexed: {stats['documents']}")
    print(f"Unique tokens: {stats['unique_tokens']}")
    print(f"Index size (KB): {stats['index_size_kb']}")
    print(f"Partial indexes: {stats['partial_indexes_written']}")
    print(f"Duplicate URLs: {stats['duplicate_urls_skipped']}")
    print(f"Near-duplicates: {stats['near_duplicates_skipped']}")
    print(f"Index saved to: {args.output}")

