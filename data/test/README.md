# Held-out test split

No test items are stored in this repository.

`mooresense generate --split test` writes `data/private/test.jsonl`, which
is gitignored. A later release can zip that file together with `CANARY.txt`.
Do not commit the zip or the JSONL. The public split is `data/dev.jsonl`.
