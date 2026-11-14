import json
import math
import sqlite3
from pathlib import Path

from link_analysis import load_graph, compute_hits


def load_extras(index_dir, document_count):
    directory = Path(index_dir)
    files = [directory / 'link_graph.jsonl', directory / 'pagerank.json', directory / 'anchor_index.sqlite3']
    if not all(p.is_file() for p in files):
        return None
    graph = load_graph(files[0], document_count)
    pagerank = {int(k): float(v) for k, v in json.loads(files[1].read_text()).items()}
    return {'graph': graph, 'pagerank': pagerank, 'anchor_db': files[2], 'n': document_count}


def rerank_with_links(query_tokens, ranked_results, extras):
    if not ranked_results or extras is None:
        return ranked_results
    candidates = {doc_id for doc_id, _ in ranked_results}
    # hits from the top matching pages and their links
    hits = compute_hits(extras['graph'], [doc for doc, _ in ranked_results[:200]])
    anchor = {}
    terms = list(dict.fromkeys(query_tokens))
    if terms:
        marks = ','.join('?' for _ in terms)
        with sqlite3.connect(extras['anchor_db']) as db:
            rows = db.execute(f'SELECT token, doc_id, tf FROM anchors WHERE token IN ({marks})', terms)
            for token, doc_id, tf in rows:
                if doc_id in candidates:
                    anchor[doc_id] = anchor.get(doc_id, 0.0) + min(3.0, math.log1p(tf))
    scores = []
    for doc_id, content_score in ranked_results:
        rank = extras['pagerank'].get(doc_id, 0.0)
        pr_boost = 1.5 * math.log1p(rank * extras['n'])
        authority_boost = 2.0 * hits.get(doc_id, 0.0)
        anchor_boost = 1.5 * anchor.get(doc_id, 0.0)
        scores.append((doc_id, content_score + pr_boost + authority_boost + anchor_boost))
    return sorted(scores, key=lambda item: (-item[1], item[0]))
