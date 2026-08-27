with open(r'd:\AI Dev\ll-fintuning\ll-finetuning\data\comprehensive_cybersec_finetune.jsonl', 'r', encoding='utf-8') as f:
    content = f.read()

# The file seems to have the first object incomplete (missing closing brace and metadata)
# Let's find where the first object should end and the second begins
# The second object starts with {"instruction":
idx = content.find('{"instruction":', 1)
print('Second object starts at:', idx)
if idx > 0:
    print('First object end context:', repr(content[idx-100:idx]))
else:
    print('Not found')