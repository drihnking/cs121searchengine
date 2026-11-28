"""Disk-backed anchor-text inverted index. No database dependencies.

The writer uses sorted partial postings and a streaming merge so the complete
anchor index does not need to be resident in memory.
"""
import heapq
import json
from pathlib import Path


def flush_anchor_chunk(counts, partial_dir, number):
    """Write sorted (term, target document, count) rows to a partial file."""
    path = partial_dir / f'anchors_{number:05d}.jsonl'
    with path.open('w', encoding='utf-8') as output:
        for (token, doc_id), tf in sorted(counts.items()):
            output.write(json.dumps([token, doc_id, tf], ensure_ascii=False, separators=(',', ':')) + '\n')
    counts.clear()
    return path


def _iter_partial(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            token, doc_id, tf = json.loads(line)
            yield token, int(doc_id), int(tf)


def merge_anchor_chunks(partial_paths, output_dir):
    """Merge partials into anchor_index.jsonl and a JSON token-offset lexicon.

    Offset positions are byte offsets, compatible with binary seek()/readline().
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / 'anchor_index.jsonl'
    lexicon_path = output_dir / 'anchor_lexicon.json'
    offsets = {}
    posting_count = 0

    with index_path.open('wb') as target:
        current_token = None
        postings = []
        last_doc = None
        last_tf = 0

        def emit_doc():
            nonlocal last_doc, last_tf, posting_count
            if last_doc is not None:
                postings.append([last_doc, last_tf])
                posting_count += 1

        def emit_token():
            if current_token is not None:
                offsets[current_token] = target.tell()
                record = {'token': current_token, 'postings': postings}
                target.write((json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8'))

        for token, doc_id, tf in heapq.merge(*(_iter_partial(Path(p)) for p in partial_paths)):
            if token != current_token:
                emit_doc()
                emit_token()
                current_token, postings, last_doc, last_tf = token, [], doc_id, tf
            elif doc_id != last_doc:
                emit_doc()
                last_doc, last_tf = doc_id, tf
            else:
                last_tf += tf
        emit_doc()
        emit_token()

    lexicon_path.write_text(json.dumps(offsets, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    return {'anchor_postings': posting_count, 'anchor_terms': len(offsets),
            'anchor_index_path': str(index_path), 'anchor_lexicon_path': str(lexicon_path)}


def read_anchor_postings(index_path, lexicon, token):
    """Read only the requested token's postings using a byte offset."""
    offset = lexicon.get(token)
    if offset is None:
        return []
    with Path(index_path).open('rb') as stream:
        stream.seek(int(offset))
        line = stream.readline()
    if not line:
        raise ValueError(f'Invalid anchor lexicon offset for {token!r}')
    record = json.loads(line)
    if record['token'] != token:
        raise ValueError(f'Anchor lexicon mismatch for {token!r}')
    return [(int(doc_id), int(tf)) for doc_id, tf in record['postings']]
