import re
tex = open('paper_elmar/em_gaussian_splatting.tex', encoding='utf-8').read()

refs   = set(re.findall(r'\\ref\{([^}]+)\}', tex))
labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
missing = refs - labels
print("Missing labels (\\ref without \\label):")
for r in sorted(missing):
    print(f"  [{r}]")
if not missing:
    print("  (none — all cross-refs resolved)")

secs  = re.findall(r'((?:sub)*section)\*?\{([^}]+)\}', tex)
print("\nSection structure:")
for kind, name in secs:
    indent = "  " * kind.count("sub")
    print(f"  {indent}\\{kind}: {name}")

# Check no placeholder XX still present
placeholders = [(m.start(), m.group()) for m in re.finditer(r'XX\.XX|\\textbf\{XX', tex)]
print("\nRemaining XX placeholders:", len(placeholders))
for pos, txt in placeholders:
    line = tex[:pos].count('\n') + 1
    print(f"  line {line}: {txt}")
