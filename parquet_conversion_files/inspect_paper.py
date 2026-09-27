import zipfile, json
z = zipfile.ZipFile('archive (4).zip')
all_keys = set()
nested_keys = set()
with z.open('paper.json') as f:
    for i in range(5000):
        line = f.readline()
        if not line:
            break
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict):
            all_keys.update(rec.keys())
            for k, v in rec.items():
                if isinstance(v, (list, dict)):
                    nested_keys.add(k)
print('total distinct keys in 5000 records:', len(all_keys))
print('keys with nested (list/dict) values:', sorted(nested_keys))
print('all keys:', sorted(all_keys))
# show one full record
with z.open('paper.json') as f:
    rec = json.loads(f.readline())
print('\nsample record keys + value types:')
for k, v in rec.items():
    t = type(v).__name__
    preview = str(v)[:60]
    print(f'  {k}: {t} = {preview}')
