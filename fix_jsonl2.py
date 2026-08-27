import json

with open(r'd:\AI Dev\ll-fintuning\ll-finetuning\data\comprehensive_cybersec_finetune.jsonl', 'r', encoding='utf-8') as f:
    content = f.read()

# The file has objects separated by newlines, but each object is split across 2 lines
# Line 1: main object (incomplete - missing closing })
# Line 2: metadata object
# Line 3: next main object (incomplete)
# Line 4: next metadata object
# etc.

# Let's parse by finding complete JSON objects
decoder = json.JSONDecoder()
objects = []
idx = 0
while idx < len(content):
    # Skip whitespace
    while idx < len(content) and content[idx].isspace():
        idx += 1
    if idx >= len(content):
        break
    try:
        obj, end = decoder.raw_decode(content, idx)
        objects.append(obj)
        idx = end
    except json.JSONDecodeError as e:
        print('Error at position {}: {}'.format(idx, e))
        # Try to find next {
        next_brace = content.find('{', idx + 1)
        if next_brace == -1:
            break
        idx = next_brace

print('Parsed {} objects'.format(len(objects)))
for i, obj in enumerate(objects):
    has_meta = 'metadata' in obj
    print('  Object {}: keys={}, has_metadata={}'.format(i+1, list(obj.keys()), has_meta))

# Now write back as proper JSONL
with open(r'd:\AI Dev\ll-fintuning\ll-finetuning\data\comprehensive_cybersec_finetune.jsonl', 'w', encoding='utf-8') as f:
    for obj in objects:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

print('File rewritten successfully')