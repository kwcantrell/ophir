#!/usr/bin/env bash
# search — query local SearXNG instance
# usage: search your query here
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: search your query here"
    exit 1
fi

QUERY=$(python3 -c "
import urllib.parse, sys
print(urllib.parse.quote(' '.join(sys.argv[1:])))" "$@")

SEARXNG="${SEARXNG_URL:-http://host.docker.internal:80}"

curl -s "${SEARXNG}/search?q=${QUERY}&format=json" \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
results = d.get('results', [])
engines = set(r.get('engine','') for r in results)
engines_str = ', '.join(engines)
print(f'=== {len(results)} results ===')
print(f'Engines: {engines_str}')
print()
for i, r in enumerate(results[:5], 1):
    print(f'{i}. {r.get(\"title\",\"\")}')
    print(f'   {r.get(\"url\",\"\")}')
    c = r.get('content','')[:300]
    if c:
        print(f'   {c}')
    print()
"