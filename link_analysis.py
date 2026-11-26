"""build link data from the downloaded pages
"""
import argparse
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup
from main import tokenize_and_stem
from anchor_storage import flush_anchor_chunk, merge_anchor_chunks


def canonical_url(url):
    """clean up a url so we can match it to an indexed page."""
    if not isinstance(url, str) or not url.strip():
        return ''
    try:
        parts = urlsplit(url.strip())
        if parts.scheme.lower() not in ('http', 'https') or not parts.netloc:
            return ''
        scheme, host = parts.scheme.lower(), parts.netloc.lower()
        # keep the path as-is, but remove the #fragment
        path = parts.path or '/'
        return urlunsplit((scheme, host, path, parts.query, ''))
    except ValueError:
        return ''


def url_keys(url):
    """try the url with and without the ending slash"""
    key = canonical_url(url)
    if not key:
        return []
    parts = urlsplit(key)
    keys = [key]
    if parts.path not in ('', '/'):
        alternate = parts.path.rstrip('/') if parts.path.endswith('/') else parts.path + '/'
        keys.append(urlunsplit((parts.scheme, parts.netloc, alternate, parts.query, '')))
    return keys


def load_document_urls(path):
    mapping = {}
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            mapping[int(row['doc_id'])] = row['url']
    return mapping


def make_url_lookup(urls):
    lookup = {}
    for doc_id, url in urls.items():
        for key in url_keys(url):
            lookup.setdefault(key, doc_id)
    return lookup


def build_link_data(corpus, output_dir, max_links_per_page=10000,
                    flush_pages=250, flush_pairs=150000):
    corpus, output_dir = Path(corpus), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    urls = load_document_urls(output_dir / 'url_mapping.jsonl')
    lookup = make_url_lookup(urls)
    graph_path = output_dir / 'link_graph.jsonl'
    seen = set()
    total_edges = scanned = skipped = 0
    counts = Counter()
    partial_paths = []

    # Temporary partials are deleted after a successful or failed build.
    with tempfile.TemporaryDirectory(prefix='anchor_parts_') as scratch:
        partial_dir = Path(scratch)
        with graph_path.open('w', encoding='utf-8') as out:
            for root, dirs, names in os.walk(corpus):
                dirs.sort()
                for name in sorted(names):
                    if not name.lower().endswith('.json'):
                        continue
                    path = Path(root) / name
                    try:
                        with path.open(encoding='utf-8-sig') as stream:
                            page = json.load(stream)
                        if not isinstance(page, dict):
                            continue
                        url, html = page.get('url', ''), page.get('content', '')
                        if not isinstance(url, str) or not isinstance(html, str):
                            continue
                        source = next((lookup[k] for k in url_keys(url) if k in lookup), None)
                        if source is None or source in seen:
                            continue
                        # Mark as seen only after completing the page successfully.
                        soup = BeautifulSoup(html, 'html.parser')
                        destinations = set()
                        anchor_counts = Counter()
                        for anchor in soup.find_all('a', href=True, limit=max_links_per_page):
                            href = anchor.get('href', '')
                            if not isinstance(href, str):
                                continue
                            target_url = urljoin(url, href)
                            target = next((lookup[k] for k in url_keys(target_url) if k in lookup), None)
                            if target is None or target == source:
                                continue
                            destinations.add(target)
                            for token in tokenize_and_stem(anchor.get_text(' ', strip=True)[:500])[:40]:
                                anchor_counts[(token, target)] += 1
                        out.write(json.dumps({'doc_id': source, 'outlinks': sorted(destinations)}) + '\n')
                        counts.update(anchor_counts)
                        seen.add(source)
                        scanned += 1
                        total_edges += len(destinations)
                        if scanned % flush_pages == 0 or len(counts) >= flush_pairs:
                            if counts:
                                partial_paths.append(flush_anchor_chunk(counts, partial_dir, len(partial_paths)))
                        if scanned % 500 == 0:
                            print(f'Link pass: {scanned:,}/{len(urls):,} pages, {total_edges:,} edges', flush=True)
                    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
                        skipped += 1
                        print(f'Skipped {path}: {exc}', flush=True)
        if counts:
            partial_paths.append(flush_anchor_chunk(counts, partial_dir, len(partial_paths)))
        anchor_stats = merge_anchor_chunks(partial_paths, output_dir)

    # To do: benchmark peak memory and runtime on the full developer collection.
    return {'indexed_documents': len(urls), 'pages_scanned': scanned,
            'edges': total_edges, 'files_skipped': skipped, **anchor_stats}


def load_graph(path, document_count=None):
    graph = {}
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            graph[int(row['doc_id'])] = [int(x) for x in row['outlinks']]
    if document_count is not None:
        for doc_id in range(document_count):
            graph.setdefault(doc_id, [])
    return graph

def build_reverse_graph(graph):
    """Compute incoming edges ONCE when initializing the search engine."""
    reverse = defaultdict(list)
    for source, targets in graph.items():
        for target in set(targets):
            if target != source:
                reverse[target].append(source)
    return dict(reverse)

def compute_pagerank(graph, damping=0.85, iterations=50, tolerance=1e-10):
    """calculate pagerank, including pages with no outgoing links."""
    nodes = sorted(set(graph) | {v for vs in graph.values() for v in vs})
    if not nodes:
        return {}
    n = len(nodes)
    rank = {u: 1.0 / n for u in nodes}
    outgoing = {u: tuple(set(graph.get(u, ())) - {u}) for u in nodes}
    # each round passes score along outgoing links
    for _ in range(iterations):
        dangling = sum(rank[u] for u in nodes if not outgoing[u])
        base = (1 - damping) / n + damping * dangling / n
        new_rank = dict.fromkeys(nodes, base)
        for source in nodes:
            edges = outgoing[source]
            if edges:
                contribution = damping * rank[source] / len(edges)
                for target in edges:
                    new_rank[target] += contribution
        delta = sum(abs(new_rank[u] - rank[u]) for u in nodes)
        rank = new_rank
        if delta < tolerance:
            break
    return rank


def compute_hits(graph, seed_docs, max_seeds=200, max_base=2000,
                 iterations=25, reverse_graph=None):
    """HITS authority scores for query results; optional precomputed reverse graph."""
    roots = list(dict.fromkeys(seed_docs))[:max_seeds]
    if not roots:
        return {}
    roots_set = set(roots)
    reverse = reverse_graph if reverse_graph is not None else build_reverse_graph(graph)
    base = set(roots)
    # Preserve the original bounded-base heuristic; never exceed max_base.
    for source in roots:
        for neighbor in sorted(set(graph.get(source, ())) | set(reverse.get(source, ()))):
            if len(base) >= max(max_base, len(roots_set)):
                break
            base.add(neighbor)
        if len(base) >= max(max_base, len(roots_set)):
            break
    adjacency = {u: [v for v in graph.get(u, ()) if v in base and v != u] for u in base}
    authorities = dict.fromkeys(base, 1.0)
    hubs = dict.fromkeys(base, 1.0)
    for _ in range(iterations):
        a = dict.fromkeys(base, 0.0)
        for source, targets in adjacency.items():
            for target in targets:
                a[target] += hubs[source]
        norm = math.sqrt(sum(x*x for x in a.values())) or 1.0
        a = {u: x/norm for u, x in a.items()}
        h = {u: sum(a[v] for v in targets) for u, targets in adjacency.items()}
        norm = math.sqrt(sum(x*x for x in h.values())) or 1.0
        h = {u: x/norm for u, x in h.items()}
        authorities, hubs = a, h
    best = max(authorities.values(), default=0) or 1.0
    return {u: authorities.get(u, 0)/best for u in roots_set}


def build_all(corpus, output_dir):
    # build the link files first, then calculate global pagerank
    stats = build_link_data(corpus, output_dir)
    graph = load_graph(Path(output_dir) / 'link_graph.jsonl', stats['indexed_documents'])
    ranks = compute_pagerank(graph)
    with (Path(output_dir) / 'pagerank.json').open('w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in ranks.items()}, f)
    with (Path(output_dir) / 'link_stats.json').open('w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print('Link analysis complete:', stats, flush=True)
    print('PageRank total:', round(sum(ranks.values()), 8), flush=True)
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build link graph, anchor index, and PageRank without rebuilding main index')
    parser.add_argument('corpus', type=Path)
    parser.add_argument('output', nargs='?', type=Path, default=Path('index_output'))
    args = parser.parse_args()
    build_all(args.corpus, args.output)
