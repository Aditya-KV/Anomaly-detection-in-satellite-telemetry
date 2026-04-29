import os
import base64

with open('report.html', 'r', encoding='utf-8') as f:
    html = f.read()

os.makedirs('assets', exist_ok=True)
parts = html.split('data:image/png;base64,')[1:]

for i, part in enumerate(parts):
    # The base64 string ends at the first quote
    b64_str = part.split("'")[0].split('"')[0]
    with open(f'assets/plot_{i}.png', 'wb') as f:
        f.write(base64.b64decode(b64_str))
