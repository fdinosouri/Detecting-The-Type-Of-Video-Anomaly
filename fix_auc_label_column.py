from pathlib import Path

path = Path("main.py")
text = path.read_text(encoding="utf-8")

old = '''        try:
            label = int(parts[2])
        except:
            try:
                label = int(parts[-1])
            except:
                continue
'''

new = '''        try:
            label = int(parts[-1])
        except:
            continue
'''

if old not in text:
    raise RuntimeError("Target label parsing block was not found.")

text = text.replace(old, new)
path.write_text(text, encoding="utf-8")

print("DONE: AUC label column fixed.")