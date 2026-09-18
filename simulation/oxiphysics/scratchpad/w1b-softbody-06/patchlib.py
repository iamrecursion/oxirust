"""Tiny exact-match patch helper (read-modify-write, fails unless the old text is unique)."""


def patch(path, old, new, count=1):
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    found = text.count(old)
    if found != count:
        raise SystemExit(f"{path}: expected {count} occurrence(s) of the old text, found {found}:\n{old[:200]}")
    text = text.replace(old, new)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"patched {path} ({len(old)} -> {len(new)} chars)")
