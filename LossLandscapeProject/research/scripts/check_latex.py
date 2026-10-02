"""Structural sanity check of paper/main.tex without a TeX engine:
balanced environments and braces, even number of $, figures exist, refs/labels/cites consistent."""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = open(os.path.join(ROOT, "paper", "main.tex"), encoding="utf-8").read()
bib = open(os.path.join(ROOT, "paper", "refs.bib"), encoding="utf-8").read()
body = re.sub(r"(?<!\\)%.*", "", s)
ok = True

stack = []
for m in re.finditer(r"\\(begin|end)\{([^}]*)\}", body):
    kind, env = m.groups()
    line = body[:m.start()].count("\n") + 1
    if kind == "begin":
        stack.append((env, line))
    elif not stack or stack[-1][0] != env:
        print(f"MISMATCH \\end{{{env}}} at line {line}; open: {stack[-1] if stack else None}")
        ok = False
        break
    else:
        stack.pop()
if stack:
    print("unclosed environments:", stack)
    ok = False

depth, line = 0, 1
for i, ch in enumerate(body):
    if ch == "\n":
        line += 1
    escaped = i > 0 and body[i - 1] == "\\"
    if ch == "{" and not escaped:
        depth += 1
    elif ch == "}" and not escaped:
        depth -= 1
        if depth < 0:
            print("extra } at line", line)
            ok = False
            break
if depth != 0:
    print("unbalanced braces, final depth", depth)
    ok = False

if len(re.findall(r"(?<!\\)\$", body)) % 2:
    print("odd number of $")
    ok = False

for f in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", body):
    if not os.path.exists(os.path.join(ROOT, "figures", f)):
        print("missing figure", f)
        ok = False

labels = set(re.findall(r"\\label\{([^}]*)\}", body))
for r in set(re.findall(r"\\(?:ref|eqref)\{([^}]*)\}", body)) - labels:
    print("undefined ref", r)
    ok = False
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cited = set(k.strip() for grp in re.findall(r"\\cite\w*\{([^}]*)\}", body) for k in grp.split(","))
for k in cited - keys:
    print("undefined citation", k)
    ok = False
print("uncited bib entries:", sorted(keys - cited))
print("pending markers:", len(re.findall(r"\\pending\{", body)) - 0)
print("OK" if ok else "PROBLEMS FOUND")
sys.exit(0 if ok else 1)
