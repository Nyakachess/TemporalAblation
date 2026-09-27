import zipfile
z = zipfile.ZipFile('archive (4).zip')
with z.open('paper.json') as f:
    biggest = 0
    for i in range(100000):
        line = f.readline()
        if not line:
            print('EOF at line', i)
            break
        biggest = max(biggest, len(line))
        if i < 5 or len(line) > 10_000_000:
            print(f'line {i}: {len(line):,} bytes')
    print(f'biggest line in sample: {biggest:,} bytes')
