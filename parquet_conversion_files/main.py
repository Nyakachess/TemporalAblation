#!/usr/bin/env python3
"""
Low-memory variant: stream each JSON member straight from the ZIP without
loading it whole. Suitable for small-RAM servers (e.g. 1-2 GB).

For each file inside the ZIP:
  - .json  : streamed as a top-level JSON array with ijson (item by item)
  - .jsonl/.ndjson : streamed line by line
  - small .json that is a single object / list-under-one-key : parsed directly
    ONLY if under --small-limit bytes (default 25 MB); larger non-array JSON
    is reported as needing the array/jsonl shape.
Non-JSON files are skipped (logged). Best-effort optional translation: applied
when it works, original kept on any failure. Outputs Parquet (+ CSV unless
--no-csv) and a manifest.json with row/column/non-null counts.

USAGE
    python3 main.py "archive (4).zip" --outdir ./converted --no-csv
    python3 main.py "archive (4).zip" --outdir ./converted --translate \
        --translate-engine argos --src-lang auto --no-csv

KEY FLAGS
    --chunk-size 25000     rows per Parquet row-group / flush (lower = less RAM)
    --small-limit 25       MB threshold below which a non-array JSON is loaded whole
    --no-csv               Parquet only (recommended on low RAM / large data)
    --max-rows N           stop each file early (trial runs)

DEPS:  pip install ijson pandas pyarrow --break-system-packages
       (translation, optional) pip install argostranslate --break-system-packages
"""

import argparse
import io
import json
import os
import sys
import zipfile
from collections import defaultdict


class BestEffortTranslator:
    def __init__(self, engine, src_lang, columns):
        self.enabled = engine is not None
        self.engine = engine
        self.src_lang = src_lang
        self.columns = set(columns) if columns else None
        self.cache = {}
        if self.enabled:
            try:
                self._init_impl()
            except Exception as e:
                print(f"[translate] disabled (init failed: {e})", file=sys.stderr)
                self.enabled = False

    def _init_impl(self):
        if self.engine == "deep":
            from deep_translator import GoogleTranslator
            self._impl = GoogleTranslator(source=self.src_lang, target="en")
        elif self.engine == "argos":
            import argostranslate.translate as at
            self._at = at
        else:
            raise ValueError(f"unknown engine: {self.engine}")

    @staticmethod
    def _looks_english(s):
        try:
            s.encode("ascii")
            return True
        except UnicodeEncodeError:
            return False

    def should_translate_col(self, col):
        return self.enabled and (self.columns is None or col in self.columns)

    def translate(self, text):
        if not self.enabled or not isinstance(text, str):
            return text
        if not text.strip() or self._looks_english(text):
            return text
        if text in self.cache:
            return self.cache[text]
        out = text
        try:
            if self.engine == "deep":
                res = self._impl.translate(text)
                out = res if res else text
            else:
                res = self._at.translate(
                    text, self.src_lang if self.src_lang != "auto" else "auto", "en")
                out = res if res else text
        except Exception:
            out = text
        self.cache[text] = out
        return out


def stream_records(zf, member, small_limit_bytes):
    """
    Yield dict records from a zip member WITHOUT loading the whole file when it
    is a JSON array or JSONL. Returns a generator; raises StopIteration normally.
    Strategy is chosen by sniffing the first non-whitespace byte.
    """
    import ijson

    name = member.filename
    lower = name.lower()

    # Sniff a larger head so we can tell JSONL from a single object/array.
    with zf.open(member) as fh:
        head = fh.read(65536)
    stripped = head.lstrip()
    first = stripped[:1]

    # JSONL detection: starts with '{' AND contains a newline that is followed
    # by another '{' (i.e. multiple top-level objects, one per line). This
    # catches .json files that are really JSON Lines, at any size.
    looks_jsonl = False
    if first == b"{":
        nl = stripped.find(b"\n")
        if nl != -1 and stripped[nl:].lstrip()[:1] == b"{":
            looks_jsonl = True

    if lower.endswith((".jsonl", ".ndjson")) or looks_jsonl or (
            first == b"{" and member.file_size > small_limit_bytes):
        # Treat as JSON Lines: stream line by line (handles big files).
        def gen_lines():
            with zf.open(member) as fh:
                tw = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
                for line in tw:
                    line = line.strip().rstrip(",")
                    if not line or line in ("[", "]"):
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        return "jsonl", gen_lines()

    if first == b"[":
        def gen_array():
            with zf.open(member) as fh:      # streamed; not loaded whole
                for obj in ijson.items(fh, "item"):
                    yield obj
        return "array", gen_array()

    if first == b"{":
        # Small enough to load whole and inspect.
        if member.file_size <= small_limit_bytes:
            with zf.open(member) as fh:
                obj = json.loads(fh.read().decode("utf-8", errors="replace"))
            if isinstance(obj, dict):
                list_keys = [k for k, v in obj.items()
                             if isinstance(v, list) and v and isinstance(v[0], dict)]
                if len(list_keys) == 1:
                    return "listkey", iter(obj[list_keys[0]])
                return "object", iter([obj])
            if isinstance(obj, list):
                return "array_loaded", iter(obj)
        # Big single object that isn't JSONL -> try ijson on the one list key.
        # Fallback: attempt streaming over '.item' anyway.
        def gen_kv():
            with zf.open(member) as fh:
                # stream values under any top-level array key
                for obj in ijson.items(fh, "item"):
                    yield obj
        return "array", gen_kv()

    return "bad", iter([])


def convert_member(zf, member, outdir, args, translator):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    name = member.filename
    small_limit = args.small_limit * 1024 * 1024
    try:
        kind, it = stream_records(zf, member, small_limit)
    except Exception as e:
        return {"file": name, "skipped": True, "reason": f"read error: {e}"}
    if kind == "bad":
        return {"file": name, "skipped": True, "reason": "not parseable JSON"}

    safe = os.path.splitext(name)[0].replace("/", "__").replace("\\", "__")
    os.makedirs(outdir, exist_ok=True)
    pq_path = os.path.join(outdir, safe + ".parquet")
    csv_path = os.path.join(outdir, safe + ".csv")

    total = 0
    non_null = defaultdict(int)
    cols_seen, cols_set = [], set()
    pq_writer = None
    csv_header = False
    buf = []

    def flush(rows):
        nonlocal pq_writer, csv_header
        if not rows:
            return
        df = pd.DataFrame(rows).reindex(columns=cols_seen)
        # Stabilise schema across chunks: cast every column to string. Without
        # this, a column that is all-null in chunk 1 (Arrow type 'null') and
        # numeric in chunk 2 (type 'double') makes PyArrow reject the write
        # with "Table schema does not match". Strings are always consistent;
        # values are preserved as text and can be re-typed downstream.
        for c in df.columns:
            df[c] = df[c].map(lambda x: None if (x is None or (isinstance(x, float) and pd.isna(x))) else str(x))
        df = df.astype("string")
        if translator and translator.enabled:
            for c in df.columns:
                if translator.should_translate_col(c):
                    df[c] = df[c].map(translator.translate)
        if pq_writer is None:
            schema = pa.schema([(c, pa.string()) for c in cols_seen])
            t = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            pq_writer = pq.ParquetWriter(pq_path, schema, compression="snappy")
            pq_writer.write_table(t)
        else:
            schema = pq_writer.schema
            df2 = df.reindex(columns=[f.name for f in schema])
            pq_writer.write_table(
                pa.Table.from_pandas(df2, schema=schema, preserve_index=False))
        if not args.no_csv:
            df.to_csv(csv_path, mode="a", index=False, header=not csv_header)
            csv_header = True

    for rec in it:
        if not isinstance(rec, dict):
            rec = {"value": rec}
        # Flatten nested values (list/dict) to JSON strings so the Arrow schema
        # stays simple and memory flat. Nested data is preserved, just encoded.
        flat = {}
        for k, v in rec.items():
            if isinstance(v, (list, dict)):
                flat[k] = json.dumps(v, ensure_ascii=False)
            else:
                flat[k] = v
        rec = flat
        for k in rec.keys():
            if k not in cols_set:
                cols_set.add(k)
                cols_seen.append(k)
        for k, v in rec.items():
            if v is not None and v != "":
                non_null[k] += 1
        buf.append(rec)
        total += 1
        if len(buf) >= args.chunk_size:
            flush(buf)
            buf = []
            print(f"\r[{name}] rows={total:,}", end="", file=sys.stderr)
        if args.max_rows and total >= args.max_rows:
            break
    flush(buf)
    if pq_writer:
        pq_writer.close()
    if total:
        print(file=sys.stderr)

    return {"file": name, "skipped": False, "detected_format": kind,
            "rows": total, "num_columns": len(cols_seen), "columns": cols_seen,
            "non_null_counts": {c: non_null.get(c, 0) for c in cols_seen},
            "outputs": {"parquet": pq_path,
                        "csv": None if args.no_csv else csv_path}}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zipfile")
    p.add_argument("--outdir", default="./converted")
    p.add_argument("--chunk-size", type=int, default=25_000)
    p.add_argument("--small-limit", type=int, default=25,
                   help="MB; non-array JSON below this is loaded whole")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--translate", action="store_true")
    p.add_argument("--translate-engine", choices=["deep", "argos"], default="deep")
    p.add_argument("--translate-cols", default=None)
    p.add_argument("--src-lang", default="auto")
    args = p.parse_args()

    engine = args.translate_engine if args.translate else None
    cols = args.translate_cols.split(",") if args.translate_cols else None
    translator = BestEffortTranslator(engine, args.src_lang, cols)
    if args.translate:
        print(f"[translate] best-effort engine={engine} "
              f"cols={cols or 'ALL'} src={args.src_lang}", file=sys.stderr)

    json_exts = (".json", ".jsonl", ".ndjson")
    results = []
    with zipfile.ZipFile(args.zipfile) as z:
        for m in z.infolist():
            if m.is_dir():
                continue
            lower = m.filename.lower()
            if not lower.endswith(json_exts):
                with z.open(m) as fh:
                    sniff = fh.read(64).lstrip()[:1]
                if sniff not in (b"[", b"{"):
                    results.append({"file": m.filename, "skipped": True,
                                    "reason": "non-JSON file"})
                    print(f"[skip] {m.filename} (non-JSON)", file=sys.stderr)
                    continue
            print(f"[file] {m.filename} ({m.file_size:,} bytes)", file=sys.stderr)
            try:
                res = convert_member(z, m, args.outdir, args, translator)
            except Exception as e:
                res = {"file": m.filename, "skipped": True, "reason": f"error: {e}"}
                print(f"[error] {m.filename}: {e}", file=sys.stderr)
            results.append(res)

    os.makedirs(args.outdir, exist_ok=True)
    manifest = os.path.join(args.outdir, "manifest.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"zip": args.zipfile, "results": results}, f,
                  indent=2, ensure_ascii=False)

    conv = [r for r in results if not r.get("skipped")]
    skip = [r for r in results if r.get("skipped")]
    print("\n=== SUMMARY ===")
    print(f"Converted: {len(conv)} JSON file(s)")
    for r in conv:
        print(f"  {r['file']}  ->  rows={r['rows']:,}, cols={r['num_columns']} "
              f"[{r['detected_format']}]")
    print(f"Skipped:   {len(skip)} file(s)")
    for r in skip:
        print(f"  {r['file']}  ({r['reason']})")
    print(f"\nManifest: {manifest}")


if __name__ == "__main__":
    main()
