import json
import math
from pathlib import Path
from anchor_storage import read_anchor_postings
from link_analysis import load_graph, compute_hits, build_reverse_graph


def load_extras(index_dir, document_count):
    directory = Path(index_dir)
    paths = {name: directory / name for name in (
        'link_graph.jsonl', 'pagerank.json', 'anchor_index.jsonl', 'anchor_lexicon.json')}
    if not all(path.is_file() for path in paths.values()):
        return None
    graph = load_graph(paths['link_graph.jsonl'], document_count)
    pagerank = {int(k): float(v) for k, v in json.loads(paths['pagerank.json'].read_text(encoding='utf-8')).items()}
    lexicon = json.loads(paths['anchor_lexicon.json'].read_text(encoding='utf-8'))
    return {'graph': graph, 'reverse_graph': build_reverse_graph(graph),
            'pagerank': pagerank, 'anchor_path': paths['anchor_index.jsonl'],
            'anchor_lexicon': lexicon, 'n': document_count}

def rerank_with_links(query_tokens, ranked_results, extras):
    if not ranked_results or extras is None:
        return ranked_results
    candidates = {doc_id for doc_id, _ in ranked_results}
    hits = compute_hits(extras['graph'], [doc for doc, _ in ranked_results[:200]],
                        reverse_graph=extras['reverse_graph'])
    anchor = {}
    for token in dict.fromkeys(query_tokens):
        for doc_id, tf in read_anchor_postings(extras['anchor_path'], extras['anchor_lexicon'], token):
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
