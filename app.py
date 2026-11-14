"""simple browser search page using python's built-in http server

run: python app.py index_output --host 127.0.0.1 --port 8080
open http://127.0.0.1:8080/ or use openlab port forwarding
"""

import argparse
import html
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from search import load_index, load_url_mapping, execute_query
from search_extras import load_extras


# styling for the search page and results
CSS = '''
*{box-sizing:border-box}body{margin:0;background:#f5f7fb;color:#192338;font-family:system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:850px;margin:72px auto;padding:0 24px}h1{font-size:2.2rem;margin:0 0 10px}
sub,p{color:#59687f}form{display:flex;gap:10px;margin:28px 0 18px}input{flex:1;padding:14px 17px;font:inherit;border:1px solid #c8d1e0;border-radius:12px}
button{background:#175cd3;color:white;border:0;border-radius:12px;padding:12px 22px;font:inherit;cursor:pointer}
article{padding:18px 20px;background:white;border:1px solid #e1e5ec;border-radius:13px;margin:12px 0;overflow-wrap:anywhere}
a{font-size:1.05rem;color:#124db0;text-decoration:none}a:hover{text-decoration:underline}.meta{font-size:.85rem;color:#627186;margin-top:7px}
footer{font-size:.85rem;color:#69778f;margin-top:25px}@media(max-width:540px){form{flex-direction:column}main{margin:36px auto}}
'''


# builds the html page with the search results
def render(query, results, match_count, latency_ms, extras_loaded):
    q = html.escape(query, quote=True)
    cards = ''

    # make a result card for each matching url
    for i, (url, score) in enumerate(results, 1):
        safe = html.escape(url, quote=True)
        cards += f'<article><div class="meta">Result {i}</div><a href="{safe}" target="_blank" rel="noopener noreferrer">{safe}</a><div class="meta">Relevance: {score:.4f}</div></article>'

    if query and not results:
        cards = '<p>No matching documents found.</p>'

    # show the number of matches and search time
    info = f'{match_count:,} matches · {latency_ms:.1f} ms' if query else 'Enter a search query to get started.'

    # show which ranking features are available
    features = 'TF-IDF · exact phrases · n-grams · PageRank · HITS · anchor text' if extras_loaded else 'TF-IDF · exact phrases · n-grams (build link data to enable PageRank, HITS and anchors)'

    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>UCI Crawling's Search Engine</title><style>{CSS}</style></head><body><main><h1>UCI Search Engine</h1><p>Explore more of UCI. Start here.</p><form action="" method="get"><input name="q" aria-label="Search query" placeholder="Search for UCI research, faculty, programs…" value="{q}" autofocus required><button type="submit">Search</button></form><div class="meta">{html.escape(info)}</div>{cards}<footer>{html.escape(features)}</footer></main></body></html>'''


# loads the index files and starts the search server
def run_server(index_dir, host, port):
    directory = Path(index_dir)

    print('Loading searchable index (large lexicon; startup can take time)...', flush=True)

    index = load_index(directory)
    urls = load_url_mapping(directory / 'url_mapping.jsonl')
    stats = json.loads((directory / 'index_stats.json').read_text())
    total_docs = stats['documents']

    # load extra ranking data if it was already built
    extras = load_extras(directory, total_docs)

    print(f"Ready: {len(urls):,} URLs; link ranking: {'enabled' if extras else 'not built'}", flush=True)

    # handles requests sent from the browser
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parts = urlsplit(self.path)

            # only allow the home and search pages
            if parts.path not in ('/', '/search'):
                self.send_error(404)
                return

            # get the search query from the url
            query = parse_qs(parts.query).get('q', [''])[0][:200].strip()
            results, matches, ms = [], 0, 0.0

            if query:
                start = time.perf_counter()

                # run the search and get the ranked documents
                matches, ranked = execute_query(query, index, urls, total_docs, extras)
                results = [(urls.get(str(doc_id), 'URL unavailable'), score) for doc_id, score in ranked]

                # calculate how long the search took
                ms = (time.perf_counter() - start) * 1000

            # turn the results into an html response
            body = render(query, results, matches, ms, extras is not None).encode('utf-8')

            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

    # start the server and keep it running
    server = ThreadingHTTPServer((host, port), Handler)
    print(f'Open http://{host}:{port}/', flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # close the server when it stops
        server.server_close()


# read command line options and launch the server
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('index_dir', nargs='?', default='index_output')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    run_server(args.index_dir, args.host, args.port)
