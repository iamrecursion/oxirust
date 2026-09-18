#!/usr/bin/env python3
"""Updates the code snapshots stored in this repository from the sources in `config.json`.

TAILORED VARIANT: "highest version branch".
This copy differs from the general update.py in exactly one respect: what an entry with no `ref`
follows. There it is the remote's default branch. Here it is the branch with the HIGHEST VERSION
NUMBER for a name ("0.4.3" beats "0.4.2" beats "0.4.2-rc.1"), because the upstreams on this shelf
work on a branch named for the release in progress, ahead of their default branch, and open a new
one at every release. Search for HIGHEST to find every place the behaviour differs.

Every source is fetched with plain `git` (a depth-1 fetch of one ref, plus submodules), so any
host git can reach works the same way: GitHub, GitLab, Codeberg/Forgejo, sourcehut, a bare repo
over SSH, a local path. The `.git` data is stripped afterwards, leaving only the tree at HEAD.

Requires python >= 3.12 and `git` on the system path.

CONFIG
------
The config is a tree of folders. A JSON object is a folder (key = folder name), a JSON list is
the repositories that live in that folder, and the reserved key "." means "this folder itself".
So the old `{ "category": [repos...] }` layout is still valid as it stands, and:

    {
      ".":      [ { "url": "https://codeberg.org/someone/top-level-thing" } ],
      "data":   [ { "name": "cgmath", "url": "https://github.com/rustgd/cgmath" } ],
      "lang":   { "rust": [ { "url": "https://github.com/rust-lang/cargo" } ],
                  ".":    [ { "url": "https://git.sr.ht/~someone/lives-directly-in-lang" } ] }
    }

Repository fields:
    url         (required) anything `git fetch` accepts
    name        folder name; defaults to the last URL segment without `.git`
    ref         HIGHEST: when absent, the branch whose name is the highest version number is
                followed, re-chosen on every run; if the remote has no version-named branch,
                its default branch is used and the result line says so. Otherwise a branch,
                tag, or full commit hash as usual; "HEAD" follows the default branch on purpose.
                `defaultBranch` is accepted as an alias.
                The value "manual" keeps its old meaning: skip, and remind at the end.
    submodules  default true (all of them, recursively); false for none; or choose by path:
                    { "exclude": ["src/llvm-project"] }   everything except these
                    { "only": ["src/tools/*"] }           just these (and what is inside them)
                Patterns are globs against the submodule's path from the snapshot root, and
                `*` crosses `/`. A pattern that matches nothing is reported, to catch typos.
    lfs         default true: LFS content is downloaded (needs `git-lfs` installed; without it
                git leaves pointer files, and the script says so). Set false to keep pointers.
    paths       PATHS: keep only part of a snapshot. The whole repository is still fetched;
                the filter is applied before the snapshot is moved into place.
                    { "exclude": ["LayoutTests", "Websites"] }   everything except these
                    { "only": ["Source", "LICENSE"] }            just these
                Globs against paths from the snapshot root, same rules as `submodules`. A
                pattern selects a path and everything beneath it. With `only`, nothing else
                is kept, top-level files included. Submodules under a removed path are not
                fetched at all. Patterns that match nothing are reported.

MANIFEST
--------
`manifest.json` (next to the config) records, per snapshot, the URL, ref, and exact commit it was
taken at. It makes each snapshot traceable, and lets a re-run skip anything whose remote has not
moved (one cheap `git ls-remote` per repo).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

# Looked for in this order when --config is not given. JSON is valid JSONC, so both load alike.
DEFAULT_CONFIG_NAMES = ("config.jsonc", "config.json")
MANIFEST_NAME = "manifest.json"
README_NAME = "README.md"
ARCHIVE_README_NAME = "ARCHIVE.md"  # Single-repository mode: README.md belongs to upstream
README_WIDTH = 100
STAGING_PREFIX = ".update-staging-"
HERE_KEY = "."
MANUAL_REF = "manual"
FULL_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


class ConfigError(Exception):
    pass


class GitError(Exception):
    pass


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Repo:
    path: PurePosixPath  # Where the snapshot lands, relative to the archive root
    url: str
    ref: str | None  # None = the remote's HEAD
    submodules: bool  # False = none at all
    lfs: bool
    sub_only: tuple[str, ...] = ()  # Globs; empty = no restriction
    sub_exclude: tuple[str, ...] = ()
    in_root: bool = False  # Single-repository mode: the contents land directly in the root
    path_only: tuple[str, ...] = ()  # PATHS filter: globs of what to keep; empty = everything
    path_exclude: tuple[str, ...] = ()  # PATHS filter: globs of what to leave out

    @property
    def label(self) -> str:
        return name_from_url(self.url) if self.in_root else str(self.path)

    @property
    def manual(self) -> bool:
        return self.ref == MANUAL_REF

    def settings(self) -> dict[str, Any]:
        """The part of a manifest entry that, if changed in the config, forces a re-fetch."""
        return {"url": self.url, "ref": self.ref, "submodules": self.submodule_setting(),
                "lfs": self.lfs, "paths": self.path_setting()}

    def path_setting(self) -> Any:
        """As it would be written in the config; None when no PATHS filter is in force, which is
        also what a manifest written before the filter existed reads as."""
        chosen: dict[str, list[str]] = {}
        if self.path_only:
            chosen["only"] = list(self.path_only)
        if self.path_exclude:
            chosen["exclude"] = list(self.path_exclude)
        return chosen or None

    def submodule_setting(self) -> Any:
        """As it would be written in the config: a plain bool unless a selection is in force."""
        if not self.submodules or not (self.sub_only or self.sub_exclude):
            return self.submodules
        chosen: dict[str, list[str]] = {}
        if self.sub_only:
            chosen["only"] = list(self.sub_only)
        if self.sub_exclude:
            chosen["exclude"] = list(self.sub_exclude)
        return chosen


def name_from_url(url: str) -> str:
    tail = url.rstrip("/").replace(":", "/").split("/")[-1]
    return tail.removesuffix(".git")


def check_segment(segment: str, where: str) -> None:
    if not segment or segment in (".", "..", ".git") or "/" in segment or "\\" in segment:
        raise ConfigError(f"{where}: {segment!r} is not usable as a folder name")


def parse_repo(entry: Any, folder: PurePosixPath) -> Repo:
    where = f"in folder '{folder}'"
    if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
        raise ConfigError(f"{where}: every repository needs a string 'url', got {entry!r}")

    url = entry["url"]
    name = entry.get("name") or name_from_url(url)
    check_segment(name, f"{where}, repository {url}")

    ref = entry.get("ref", entry.get("defaultBranch"))
    if ref is not None and (not isinstance(ref, str) or not ref or ref.startswith("-")):
        raise ConfigError(f"{where}, repository {name}: bad ref {ref!r}")

    setting = entry.get("submodules", True)
    only: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    if isinstance(setting, dict):
        unknown = set(setting) - {"only", "exclude"}
        if unknown:
            raise ConfigError(f"{where}, repository {name}: unknown submodules key(s) "
                              f"{sorted(unknown)}; use \"only\" and/or \"exclude\"")
        for key in ("only", "exclude"):
            patterns = setting.get(key, [])
            if isinstance(patterns, str):
                patterns = [patterns]
            if not isinstance(patterns, list) or not all(isinstance(x, str) and x for x in patterns):
                raise ConfigError(f"{where}, repository {name}: submodules.{key} must be a list "
                                  "of path globs")
            cleaned = tuple(x.strip("/") for x in patterns)
            only, exclude = (cleaned, exclude) if key == "only" else (only, cleaned)
        setting = True
    elif not isinstance(setting, bool):
        raise ConfigError(f"{where}, repository {name}: submodules must be true, false, or an "
                          "object with \"only\" and/or \"exclude\"")

    path_filter = entry.get("paths")
    path_only: tuple[str, ...] = ()
    path_exclude: tuple[str, ...] = ()
    if path_filter is not None:
        if not isinstance(path_filter, dict) or set(path_filter) - {"only", "exclude"}:
            raise ConfigError(f"{where}, repository {name}: paths must be an object with \"only\" "
                              "and/or \"exclude\"")
        for key in ("only", "exclude"):
            patterns = path_filter.get(key, [])
            if isinstance(patterns, str):
                patterns = [patterns]
            if not isinstance(patterns, list) or not all(isinstance(x, str) and x.strip("/")
                                                         for x in patterns):
                raise ConfigError(f"{where}, repository {name}: paths.{key} must be a list of "
                                  "path globs")
            cleaned = tuple(x.strip("/") for x in patterns)
            path_only, path_exclude = ((cleaned, path_exclude) if key == "only"
                                       else (path_only, cleaned))

    return Repo(
        path=folder / name,
        url=url,
        ref=ref,
        submodules=setting,
        lfs=bool(entry.get("lfs", True)),
        sub_only=only,
        sub_exclude=exclude,
        path_only=path_only,
        path_exclude=path_exclude,
    )


def parse_node(node: Any, folder: PurePosixPath) -> list[Repo]:
    if isinstance(node, list):
        return [parse_repo(entry, folder) for entry in node]

    if isinstance(node, dict):
        repos: list[Repo] = []
        for key, child in node.items():
            if key == HERE_KEY:
                if not isinstance(child, list):
                    raise ConfigError(f"in folder '{folder}': the '.' key must hold a list")
                repos += parse_node(child, folder)
            else:
                check_segment(key, f"in folder '{folder}'")
                repos += parse_node(child, folder / key)
        return repos

    raise ConfigError(f"in folder '{folder}': expected a folder object or a repository list")


def strip_jsonc(text: str) -> str:
    """Turns JSONC into JSON: removes // and /* */ comments and trailing commas.

    String-aware, so the `//` in every URL is left alone. Removed characters become spaces and
    newlines are kept, so line and column numbers in JSON error messages still point at the
    right place in the original file.
    """
    out: list[str] = []
    last_significant = -1  # Index in `out` of the last non-blank character outside comments
    i, n = 0, len(text)
    while i < n:
        char = text[i]
        if char == '"':  # Copy the whole string literal verbatim
            start = i
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
            i += 1
            out.append(text[start:i])
            last_significant = len(out) - 1
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            end = n if end == -1 else end + 2
            out.append("".join(c if c == "\n" else " " for c in text[i:end]))
            i = end
        else:
            if char in "}]" and last_significant >= 0 and out[last_significant] == ",":
                out[last_significant] = " "  # A trailing comma
            out.append(char)
            if not char.isspace():
                last_significant = len(out) - 1
            i += 1
    return "".join(out)


def load_config(config_path: Path) -> list[Repo]:
    if not config_path.exists():
        raise ConfigError(f"The provided config path {config_path} does not exist")
    raw = json.loads(strip_jsonc(config_path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ConfigError("The top level of the config must be an object")

    if "url" in raw:
        # Single-repository mode: the top-level object IS the repository, and its contents land
        # directly in the root, beside this script and its files.
        unknown = set(raw) - {"url", "ref", "defaultBranch", "submodules", "lfs", "paths"}
        if unknown:
            raise ConfigError(f"single-repository config: unexpected key(s) {sorted(unknown)} (folders "
                              "and a top-level \"url\" cannot be combined, and \"name\" has no meaning here)")
        single = parse_repo(raw | {"name": "root"}, PurePosixPath("."))
        return [replace(single, path=PurePosixPath("."), in_root=True)]

    repos = parse_node(raw, PurePosixPath("."))

    # No two snapshots may share a folder, and none may sit inside another.
    seen: dict[PurePosixPath, Repo] = {}
    for repo in repos:
        if repo.path in seen:
            raise ConfigError(f"'{repo.path}' is claimed by both {seen[repo.path].url} and {repo.url}")
        seen[repo.path] = repo
    for repo in repos:
        for parent in repo.path.parents:
            if parent in seen:
                raise ConfigError(f"'{repo.path}' sits inside the snapshot '{parent}'")
    if PurePosixPath(MANIFEST_NAME) in seen:
        raise ConfigError(f"'{MANIFEST_NAME}' is reserved for the manifest")

    return sorted(repos, key=lambda r: str(r.path))


CONFIG_HEADER = """\
// Sources for the code snapshots in this repository, read by update.py.
// This file is JSONC: // and /* */ comments and trailing commas are all fine.
//
// LAYOUT
//   An object is a folder (key = folder name). A list holds the repositories that live in that
//   folder. The key "." means "this folder itself", so the "." list at the top level holds
//   repositories that land in the root. Folders nest to any depth.
//
// REPOSITORY FIELDS
//   "url"         Required. Anything `git fetch` accepts: https, ssh, a local path; any host.
//   "name"        Folder name for the snapshot. Default: last URL segment, minus ".git".
//   "ref"         Leave it out (the normal case on this shelf) and the snapshot follows the
//                 branch whose NAME IS THE HIGHEST VERSION NUMBER: "0.4.3" over "0.4.2" over
//                 "0.4.2-rc.1"; a leading "v" is fine; branches like "master" are ignored.
//                 It is chosen afresh on every run, so when upstream opens "0.4.4" the snapshot
//                 moves to it by itself, and the result line reports the jump. If a remote has
//                 no version-named branch, its default branch is used, and reported.
//                 Or name a branch, tag, or full commit hash. "HEAD" = the default branch.
//                 A branch is followed: every run moves the snapshot to its newest tip.
//                 A tag or hash stays put. "manual" means: skip, and remind me at the end.
//                 ("defaultBranch" is accepted as an older spelling of "ref".)
//   "submodules"  Default true: all submodules are fetched, recursively. false fetches none.
//                 To choose, give globs against the submodule paths (`*` crosses `/`):
//                   { "exclude": ["src/llvm-project"] }  everything except these
//                   { "only": ["src/tools/*"] }          just these, and what is inside them
//                 A submodule nested inside an unfetched one cannot be reached, so with
//                 "only", list the parent as well. Patterns matching nothing are reported.
//   "lfs"         Default true: LFS content is downloaded (needs git-lfs installed). Set false
//                 to keep pointer files. Either way nothing is ever pushed as LFS: update.py
//                 keeps `* -filter` in this repository's info/attributes.
//   "paths"       Keep only part of a snapshot. The whole repository is still fetched; the
//                 filter is applied before the snapshot is moved into place.
//                   { "exclude": ["LayoutTests", "Websites"] }  everything except these
//                   { "only": ["Source", "LICENSE"] }           just these
//                 Globs against paths from the snapshot root, same rules as "submodules". A
//                 pattern selects a path and everything beneath it. With "only", nothing else
//                 is kept, top-level files included. Submodules under a removed path are not
//                 fetched. Patterns that match nothing are reported.
//
// EXAMPLE
//   "signals": [
//     { "url": "https://github.com/RustAudio/cpal" },
//     { "url": "https://codeberg.org/someone/thing", "name": "thing-stable", "ref": "v1.2.0" },
//   ],
//   "lang": [
//     // Everything except the one enormous submodule:
//     { "url": "https://github.com/rust-lang/rust",
//       "submodules": { "exclude": ["src/llvm-project"] } },
//     // Only the submodules under vendor/, minus one of them:
//     { "url": "https://example.org/some/project",
//       "submodules": { "only": ["vendor/*"], "exclude": ["vendor/huge-assets"] } },
//     // No submodules at all, and LFS files left as pointers:
//     { "url": "https://example.org/some/other", "submodules": false, "lfs": false },
//   ],
//
// Upstream .gitignore files are removed from every snapshot, so that `git add` keeps everything
// that came down. Upstream .gitattributes stay: line endings follow upstream.
//
// manifest.json, next to this file, is written by update.py and records the exact commit each
// snapshot was taken at. It is a record, never a pin; there is no need to edit it.
"""


CONFIG_HEADER_SINGLE = """\
// The source for the code snapshot in this repository, read by update.py.
// This file is JSONC: // and /* */ comments and trailing commas are all fine.
//
// SINGLE-REPOSITORY MODE
//   The top-level object IS the repository (it has a "url"), and the snapshot's contents land
//   directly in the root of this repository, beside update.py and its files. For an archive of
//   many repositories in folders instead, see `update.py --create-config`.
//
//   Sharing the root has rules. This archive's own files (update.py, this config, manifest.json,
//   ARCHIVE.md, .git) always win: an upstream file of the same name is skipped and reported.
//   Anything else already here that did not come from the last snapshot is left alone, and
//   reported. Only what manifest.json lists as the last snapshot's entries is ever replaced.
//
// FIELDS
//   "url"         Required. Anything `git fetch` accepts: https, ssh, a local path; any host.
//   "ref"         Leave it out (the normal case on this shelf) and the snapshot follows the
//                 branch whose NAME IS THE HIGHEST VERSION NUMBER: "0.4.3" over "0.4.2" over
//                 "0.4.2-rc.1"; a leading "v" is fine; branches like "master" are ignored.
//                 It is chosen afresh on every run, so when upstream opens "0.4.4" the snapshot
//                 moves to it by itself, and the result line reports the jump. If a remote has
//                 no version-named branch, its default branch is used, and reported.
//                 Or name a branch, tag, or full commit hash. "HEAD" = the default branch.
//                 A branch is followed: every run moves the snapshot to its newest tip.
//                 A tag or hash stays put.
//   "submodules"  Default true: all submodules are fetched, recursively. false fetches none.
//                 To choose, give globs against the submodule paths (`*` crosses `/`):
//                   { "exclude": ["thirdparty/huge"] }   everything except these
//                   { "only": ["libs/*"] }               just these, and what is inside them
//                 A submodule nested inside an unfetched one cannot be reached, so with
//                 "only", list the parent as well. Patterns matching nothing are reported.
//   "lfs"         Default true: LFS content is downloaded (needs git-lfs installed). Set false
//                 to keep pointer files. Either way nothing is ever pushed as LFS: update.py
//                 keeps `* -filter` in this repository's info/attributes.
//   "paths"       Keep only part of a snapshot. The whole repository is still fetched; the
//                 filter is applied before the snapshot is moved into place.
//                   { "exclude": ["LayoutTests", "Websites"] }  everything except these
//                   { "only": ["Source", "LICENSE"] }           just these
//                 Globs against paths from the snapshot root, same rules as "submodules". A
//                 pattern selects a path and everything beneath it. With "only", nothing else
//                 is kept, top-level files included. Submodules under a removed path are not
//                 fetched. Patterns that match nothing are reported.
//
// Upstream .gitignore files are removed from the snapshot, so that `git add` keeps everything
// that came down. Upstream .gitattributes stay: line endings follow upstream.
//
// manifest.json, next to this file, is written by update.py and records the exact commit the
// snapshot was taken at and which top-level entries it brought. It is a record, never a pin.
"""


def create_single_config(config_path: Path, source_url: str) -> None:
    """Writes a single-repository config for `source_url`. Never overwrites."""
    siblings = [config_path] + [config_path.parent / name for name in DEFAULT_CONFIG_NAMES
                                if config_path.name in DEFAULT_CONFIG_NAMES]
    for existing in siblings:
        if existing.exists():
            raise ConfigError(f"{existing} already exists; refusing to create a config beside or over it")
    if not source_url.strip():
        raise ConfigError("single-repository mode needs the URL of the repository to snapshot")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as file:
        file.write(CONFIG_HEADER_SINGLE)
        json.dump({"url": source_url.strip()}, file, indent=2)
        file.write("\n")
    print(f"Wrote {config_path}")


def refresh_config_guide(config_path: Path) -> None:
    """Replaces the guide comment at the top of an existing config with the current one.

    Only the leading block that this script generated is replaced (it is recognised by its first
    line). Anything else before the opening brace is kept, below the fresh guide; the config body
    is never touched.
    """
    if not config_path.exists():
        raise ConfigError(f"{config_path} does not exist")
    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    first_lines = {CONFIG_HEADER.splitlines()[0], CONFIG_HEADER_SINGLE.splitlines()[0]}

    body_start = next((i for i, line in enumerate(lines) if line.lstrip().startswith("{")), None)
    if body_start is None:
        raise ConfigError(f"{config_path}: could not find the opening brace")
    leading = lines[:body_start]
    if leading and leading[0].rstrip("\n") in first_lines:
        # Drop our old guide: the unbroken run of // lines from the top.
        keep_from = next((i for i, line in enumerate(leading) if not line.startswith("//")),
                         len(leading))
        leading = leading[keep_from:]

    header = CONFIG_HEADER_SINGLE if any(r.in_root for r in load_config(config_path)) else CONFIG_HEADER
    config_path.write_text(header + "".join(leading) + "".join(lines[body_start:]), encoding="utf-8")
    load_config(config_path)  # Proves the result still parses; raises if not
    print(f"Refreshed the guide at the top of {config_path}")


def create_config(config_path: Path, folder_specs: list[str]) -> None:
    """Writes a scratch config holding empty repository lists for the named folders.

    `foo,bar,baz/quux` becomes `{ ".": [], "bar": [], "baz": { "quux": [] }, "foo": [] }`: folders
    are sorted alphabetically at every level, whatever order they were given in. A folder
    that is both named itself and has children (`baz,baz/quux`) keeps its own list under ".".
    """
    siblings = [config_path] + [config_path.parent / name for name in DEFAULT_CONFIG_NAMES
                                if config_path.name in DEFAULT_CONFIG_NAMES]
    for existing in siblings:
        if existing.exists():
            raise ConfigError(f"{existing} already exists; refusing to create a config beside or over it")

    # First a tree of plain dicts, remembering which folders were named explicitly.
    tree: dict[str, Any] = {}
    named: set[tuple[str, ...]] = set()
    for spec in (part.strip() for joined in folder_specs for part in joined.split(",")):
        if not spec:
            continue
        segments = tuple(segment for segment in spec.strip("/").split("/"))
        node = tree
        for segment in segments:
            check_segment(segment, f"--create-config '{spec}'")
            node = node.setdefault(segment, {})
        named.add(segments)

    def render(node: dict[str, Any], prefix: tuple[str, ...]) -> Any:
        if not node:
            return []
        rendered: dict[str, Any] = {HERE_KEY: []} if prefix in named or not prefix else {}
        # Alphabetical at every level, ignoring case; the "." list always stays first.
        for key, child in sorted(node.items(), key=lambda item: (item[0].casefold(), item[0])):
            rendered[key] = render(child, prefix + (key,))
        return rendered

    config = render(tree, ()) or {HERE_KEY: []}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as file:
        file.write(CONFIG_HEADER)
        json.dump(config, file, indent=2)
        file.write("\n")
    print(f"Wrote {config_path}")


# --------------------------------------------------------------------------------------------
# Live status
# --------------------------------------------------------------------------------------------


class Status:
    """What every in-flight snapshot is doing right now, plus the one place that prints.

    On a terminal, a small block is redrawn in place under the scrolling result lines: an overall
    bar, then one line per active snapshot with a bar for git's current phase. The standard
    library has no progress bar, but one is only a string of blocks, so it is drawn by hand.
    When output is piped or logged, a plain heartbeat line is printed every so often instead.
    """

    HEARTBEAT_SECONDS = 30.0
    BAR_WIDTH = 24
    PERCENT = re.compile(r"(\d{1,3})%")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active: dict[str, tuple[float, str]] = {}  # snapshot -> (started, what it is doing)
        self.total = 0
        self.finished = 0
        self.tty = sys.stdout.isatty()
        self.drawn = 0  # Lines of the block currently on screen
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def set(self, snapshot: str, doing: str) -> None:
        with self.lock:
            started = self.active.get(snapshot, (time.monotonic(), ""))[0]
            self.active[snapshot] = (started, doing)

    def done(self, snapshot: str) -> None:
        with self.lock:
            self.active.pop(snapshot, None)

    @classmethod
    def bar(cls, fraction: float | None) -> str:
        if fraction is None:  # A stage git reports no percentage for
            return "·" * cls.BAR_WIDTH
        filled = round(max(0.0, min(1.0, fraction)) * cls.BAR_WIDTH)
        return "█" * filled + "░" * (cls.BAR_WIDTH - filled)

    def _snapshot(self) -> list[tuple[str, float, str]]:
        with self.lock:
            return sorted(((name, started, doing) for name, (started, doing) in self.active.items()),
                          key=lambda item: item[1])

    def block(self, width: int) -> list[str]:
        """The lines of the live block, each cut to the terminal width so that none can wrap."""
        now = time.monotonic()
        jobs = self._snapshot()
        overall = self.finished / self.total if self.total else None
        lines = [f"{self.bar(overall)} {self.finished}/{self.total} done, {len(jobs)} active"]
        name_width = max((len(name) for name, _, _ in jobs), default=0)
        for name, started, doing in jobs:
            # "fetching: Receiving objects:  45% (5120/11377), 88.2 MiB | 6.1 MiB/s"
            stage, _, report = doing.partition(": ")
            phase, _, rest = report.partition(":")
            match = self.PERCENT.search(rest)
            if match:
                percent = int(match.group(1))
                tail = rest[match.end():].strip(" ,")
                tail = tail.split("),", 1)[1].strip() if ")," in tail else ""
                label = f"{stage} · {phase.strip()}" if phase.strip() else stage
                detail = f"{self.bar(percent / 100)} {percent:>3}%  {label}" + (f"  {tail}" if tail else "")
            else:
                detail = f"{self.bar(None)}       {doing}"
            lines.append(f"{name:<{name_width}}  {detail}  [{int(now - started)}s]")
        return [line if len(line) <= width else line[: width - 1] + "…" for line in lines]

    def summary(self) -> str:
        now = time.monotonic()
        parts = [f"{name} [{int(now - started)}s] {doing}" for name, started, doing in self._snapshot()]
        return f"{self.finished}/{self.total} done, {len(parts)} active: " + " | ".join(parts)

    def _erase(self) -> None:
        """Removes the block from the screen. The cursor rests at the end of its last line."""
        if self.drawn:
            sys.stdout.write("\r" + (f"\x1b[{self.drawn - 1}A" if self.drawn > 1 else "") + "\x1b[J")
            self.drawn = 0

    def _draw(self) -> None:
        size = shutil.get_terminal_size((100, 24))
        lines = self.block(size.columns - 1)[: max(1, size.lines - 2)]
        with self.lock:
            self._erase()
            sys.stdout.write("\n".join(lines))
            sys.stdout.flush()
            self.drawn = len(lines)

    def emit(self, line: str, file: Any = None) -> None:
        """Prints a permanent line, removing the live block first so that they never collide."""
        with self.lock:
            if self.tty:
                self._erase()
                sys.stdout.flush()
            print(line, file=file or sys.stdout, flush=True)

    def _run(self) -> None:
        last_beat = time.monotonic()
        while not self.stop.wait(0.25):
            if self.tty:
                self._draw()
            elif self.active and time.monotonic() - last_beat >= self.HEARTBEAT_SECONDS:
                last_beat = time.monotonic()
                self.emit(f"  ... {self.summary()}")

    def __enter__(self) -> "Status":
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        if self.tty:
            with self.lock:
                self._erase()
                sys.stdout.flush()


STATUS = Status()


def run_watched(command: list[str], cwd: Path | None, env: dict[str, str], snapshot: str,
                stage: str) -> subprocess.CompletedProcess[str]:
    """Runs git while copying its latest progress report into STATUS.

    Git rewrites its progress in place with carriage returns, so the stream is split on both
    \\r and \\n and only the newest piece is kept for display. Everything is also kept for error
    reporting. stdout is discarded: the watched commands (fetch, submodule update) print nothing
    there that is needed, and an unread pipe could otherwise fill and stall git.
    """
    STATUS.set(snapshot, stage)
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    assert process.stderr is not None
    with LIVE_LOCK:
        LIVE.add(process)
    collected = bytearray()
    pending = b""
    while chunk := os.read(process.stderr.fileno(), 4096):
        collected += chunk
        pieces = (pending + chunk).replace(b"\r", b"\n").split(b"\n")
        pending = pieces.pop()
        latest = next((piece for piece in reversed(pieces) if piece.strip()), b"")
        if latest:
            text = latest.decode("utf-8", "replace").strip().removeprefix("remote: ")
            if cwd is not None:  # "Cloning into '<staging path>/vendor/x'" -> 'vendor/x'
                text = text.replace(f"{cwd}{os.sep}", "")
            STATUS.set(snapshot, f"{stage}: {text}")
    returncode = process.wait()
    with LIVE_LOCK:
        LIVE.discard(process)
    # Progress noise is not an error message: keep only real lines for GitError to pick from.
    stderr = "\n".join(line for line in collected.decode("utf-8", "replace").replace("\r", "\n")
                       .splitlines() if line.strip() and "%" not in line)
    return subprocess.CompletedProcess(command, returncode, "", stderr)


# --------------------------------------------------------------------------------------------
# Git
# --------------------------------------------------------------------------------------------


ABORT = threading.Event()  # Set on Ctrl-C: nothing new starts, and running git children are stopped
LIVE: set[subprocess.Popen[Any]] = set()
LIVE_LOCK = threading.Lock()


def stop_running_git() -> None:
    with LIVE_LOCK:
        for process in LIVE:
            try:
                process.terminate()
            except OSError:
                pass

_ssh_batch_mode: bool | None = None


def ssh_batch_mode_wanted() -> bool:
    """True unless the user already chose their own ssh command, which is then left alone."""
    global _ssh_batch_mode
    if _ssh_batch_mode is None:
        configured = subprocess.run(["git", "config", "--get", "core.sshCommand"],
                                    capture_output=True, text=True).stdout.strip()
        _ssh_batch_mode = not (os.environ.get("GIT_SSH_COMMAND") or os.environ.get("GIT_SSH")
                               or configured)
    return _ssh_batch_mode


def git(*args: str, cwd: Path | None = None, lfs: bool = True,
        watch: tuple[str, str] | None = None) -> str:
    """Runs git. With `watch=(snapshot, stage)` the command's own progress is streamed to the
    live status display; git must then have been given `--progress`."""
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"  # Fail on private/missing repos instead of hanging
    if ssh_batch_mode_wanted():
        # The same for ssh's own questions (unknown host key, key passphrase): with parallel
        # fetches and captured output a prompt is a hang. Agents and hardware keys still work.
        env["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes"
    if not lfs:
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
    # `core.autocrlf=false`: line endings in the scratch checkout follow upstream's own
    # .gitattributes and nothing else, whatever this machine's global git config says.
    command = ["git", "-c", "core.autocrlf=false", *args]
    if ABORT.is_set():
        raise GitError("interrupted")
    if watch is None:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        with LIVE_LOCK:
            LIVE.add(process)
        try:
            stdout, stderr = process.communicate()
        finally:
            with LIVE_LOCK:
                LIVE.discard(process)
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    else:
        result = run_watched(command, cwd, env, *watch)
    if ABORT.is_set():
        raise GitError("interrupted")
    if result.returncode != 0:
        lines = [line.strip() for line in (result.stderr or result.stdout).splitlines()
                 if line.strip()]
        # ssh and git put the cause first and boilerplate advice last, so prefer the telling lines.
        telling = [line for line in lines
                   if line.startswith(("fatal:", "error:", "ERROR:")) or "denied" in line.lower()
                   or "host key" in line.lower() or "could not resolve" in line.lower()
                   or "not found" in line.lower()]
        chosen = list(dict.fromkeys(telling or lines[-1:]))
        raise GitError(" | ".join(chosen) if chosen else f"git {args[0]} failed")
    return result.stdout.strip()


# HIGHEST: version-named branches ------------------------------------------------------------
VERSION_BRANCH = re.compile(r"^v?(\d+(?:\.\d+)*)(?:-([0-9A-Za-z.-]+))?$")


def version_key(name: str) -> tuple[Any, ...] | None:
    """A sort key for a version-like branch name, or None if the name is not a version.

    Semantic-versioning order: numeric parts compare as numbers ("0.10.0" beats "0.9.9"), and a
    pre-release sorts BEFORE the release it leads up to ("0.1.0-rc.1" loses to "0.1.0"). Within a
    pre-release, numeric identifiers compare as numbers and sort before alphabetic ones.
    """
    match = VERSION_BRANCH.match(name)
    if not match:
        return None
    numbers = tuple(int(part) for part in match.group(1).split("."))
    pre = match.group(2)
    if pre is None:
        return (numbers, 1, ())
    return (numbers, 0, tuple((0, int(part), "") if part.isdigit() else (1, 0, part)
                              for part in re.split(r"[.-]", pre)))


def highest_version_branch(url: str) -> tuple[str, str] | None:
    """(branch name, commit) of the remote's highest version-named branch, or None if it has none.
    One `ls-remote` answers both which branch and where it is, so no second round trip is needed."""
    best: tuple[tuple[Any, ...], str, str] | None = None
    for line in git("ls-remote", "--heads", url).splitlines():
        sha, _, ref = line.partition("\t")
        name = ref.removeprefix("refs/heads/")
        key = version_key(name)
        if key is not None and (best is None or key > best[0]):
            best = (key, name, sha)
    return (best[1], best[2]) if best else None


def remote_commit(repo: Repo) -> str | None:
    """The commit the configured ref currently points at on the remote, without fetching."""
    if repo.ref and FULL_SHA.match(repo.ref):
        return repo.ref.lower()
    wanted = [repo.ref, f"{repo.ref}^{{}}"] if repo.ref else ["HEAD"]
    lines = [line.split("\t") for line in git("ls-remote", repo.url, *wanted).splitlines()]
    found = {name: sha for sha, name in (line for line in lines if len(line) == 2)}
    if not found:
        return None
    for name, sha in found.items():  # An annotated tag: the peeled entry is the real commit
        if name.endswith("^{}"):
            return sha
    return found.get(f"refs/heads/{repo.ref}") or next(iter(found.values()))


NO_SHALLOW = re.compile(r"dumb http transport|does not support shallow|shallow[^|]*not supported",
                        re.IGNORECASE)


def git_shallow_or_full(before_depth: list[str], after_depth: list[str], unshallowed: list[str],
                        what: str, **kwargs: Any) -> None:
    """Runs a fetch-like git command at depth 1, and again at full depth if the server cannot
    serve shallow requests.

    Some hosts still serve git over "dumb" HTTP (plain files behind a web server, no git service
    on the other end), and that transport cannot do shallow clones at all. The history is thrown
    away after the fetch either way, so a full fetch costs only time and bandwidth.
    """
    try:
        git(*before_depth, "--depth", "1", *after_depth, **kwargs)
    except GitError as error:
        if not NO_SHALLOW.search(str(error)) or ABORT.is_set():
            raise
        unshallowed.append(what)
        git(*before_depth, *after_depth, **kwargs)


def submodule_paths(repo_dir: Path) -> list[str]:
    """The submodules that really exist at this commit: listed in `.gitmodules` AND present in the
    tree as a gitlink.

    `.gitmodules` alone is not to be trusted. Projects remove a submodule and forget its entry
    (capstone's `.gitmodules` names a tree-sitter path that its tree no longer has), and asking
    git to update such a path is an error. Git's own `--recursive` quietly goes by the gitlinks,
    so this does the same. A gitlink with no `.gitmodules` entry has no URL to fetch from, and is
    left out as well.
    """
    try:
        listing = git("config", "--file", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$",
                      cwd=repo_dir)
        staged = git("ls-files", "--stage", "-z", cwd=repo_dir)
    except GitError:
        return []
    declared = [line.split(" ", 1)[1] for line in listing.splitlines() if " " in line]
    gitlinks = {entry.split("\t", 1)[1] for entry in staged.split("\0")
                if entry.startswith("160000 ") and "\t" in entry}
    return [path for path in declared if path in gitlinks]


def fetch_chosen_submodules(repo: Repo, repo_dir: Path, prefix: PurePosixPath, inherited: bool,
                            matched: set[str], unshallowed: list[str]) -> int:
    """Walks the submodule tree by hand, so that `only` / `exclude` can apply at every level and
    so that one submodule on a server without shallow support does not fail (or un-shallow) the
    rest: each submodule gets its own depth-1 attempt and its own full-depth fallback.

    Patterns are matched against the path from the snapshot root. Once a submodule is selected by
    an `only` pattern, everything nested inside it is selected too (`inherited`), while `exclude`
    is still honoured at any depth. Returns how many submodules were left unfetched.
    """
    skipped = 0
    for path in submodule_paths(repo_dir):
        full = str(prefix / path)
        hits_only = [p for p in repo.sub_only if fnmatch.fnmatchcase(full, p)]
        hits_exclude = [p for p in repo.sub_exclude if fnmatch.fnmatchcase(full, p)]
        matched.update(hits_only, hits_exclude)

        wanted = (inherited or not repo.sub_only or bool(hits_only)) and not hits_exclude
        if wanted and not submodule_wanted_by_paths(repo, full):
            wanted = False  # PATHS: no point fetching a submodule that would be deleted again
        if not wanted:
            skipped += 1
            continue
        git_shallow_or_full(["submodule", "update", "--init"], ["--progress", "--", path],
                            unshallowed, f"submodule {full}",
                            cwd=repo_dir, lfs=repo.lfs, watch=(repo.label, f"submodule {full}"))
        child = repo_dir / path
        if (child / ".gitmodules").exists():
            skipped += fetch_chosen_submodules(repo, child, prefix / path, True, matched,
                                               unshallowed)
    return skipped


def fetch_snapshot(repo: Repo, work: Path) -> tuple[Path, str, str, list[str]]:
    """Fetches one ref at depth 1 into a fresh directory.

    Returns (dir, commit, commit date, notes worth showing the user)."""
    git("init", "--quiet", str(work))
    git("remote", "add", "origin", repo.url, cwd=work)
    unshallowed: list[str] = []
    git_shallow_or_full(["fetch"], ["--progress", "origin", repo.ref or "HEAD"], unshallowed,
                        "the repository itself", cwd=work, lfs=repo.lfs,
                        watch=(repo.label, "fetching"))
    STATUS.set(repo.label, "checking out")
    git("checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=work, lfs=repo.lfs)
    notes: list[str] = []
    if repo.submodules and (work / ".gitmodules").exists():
        matched: set[str] = set()
        skipped = fetch_chosen_submodules(repo, work, PurePosixPath("."), False, matched,
                                          unshallowed)
        if skipped:
            notes.append(f"skipped {skipped} submodule(s)")
        notes += [f"submodule pattern '{pattern}' matched nothing"
                  for pattern in repo.sub_only + repo.sub_exclude if pattern not in matched]
    if unshallowed:
        notes.append("fetched with full history, as the server cannot do shallow: "
                     + ", ".join(unshallowed))

    commit = git("rev-parse", "HEAD", cwd=work)
    commit_date = git("log", "-1", "--format=%cI", cwd=work)
    STATUS.set(repo.label, "stripping git data")
    ignores = strip_git_data(work)
    if ignores:
        notes.append(f"dropped {ignores} upstream .gitignore file(s)")
    if repo.path_only or repo.path_exclude:
        STATUS.set(repo.label, "applying the paths filter")
        notes[:0] = apply_path_filter(repo, work)
    return work, commit, commit_date, notes


# PATHS: keeping only part of a snapshot ---------------------------------------------------------


def path_hits(relative: str, patterns: tuple[str, ...]) -> list[str]:
    """The patterns that select `relative`: a pattern selects a path when it matches the path
    itself or any directory above it, so "LayoutTests" covers everything inside LayoutTests.
    Same glob rules as the submodule filter: case-sensitive, and `*` crosses `/`."""
    parts = relative.split("/")
    candidates = ["/".join(parts[:end]) for end in range(1, len(parts) + 1)]
    return [p for p in patterns if any(fnmatch.fnmatchcase(c, p) for c in candidates)]


def could_hold_wanted(relative: str, only: tuple[str, ...]) -> bool:
    """With an `only` list, whether a not-yet-selected directory could still contain something
    selected. Literal patterns settle it exactly; a pattern with a wildcard gets the benefit of
    the doubt, since emptied directories are swept up afterwards anyway."""
    return any(p.startswith(relative + "/") or any(ch in p for ch in "*?[") for p in only)


def submodule_wanted_by_paths(repo: Repo, full: str) -> bool:
    """Whether the PATHS filter leaves any reason to fetch the submodule at `full`."""
    if path_hits(full, repo.path_exclude):
        return False
    return (not repo.path_only or bool(path_hits(full, repo.path_only))
            or could_hold_wanted(full, repo.path_only))


def apply_path_filter(repo: Repo, root: Path) -> list[str]:
    """Deletes from a fetched snapshot whatever the PATHS filter says not to keep, and returns
    the notes for the result line. The whole repository is fetched first and filtered here, on
    the way in: simple, and the same on every host."""
    if not (repo.path_only or repo.path_exclude):
        return []
    matched: set[str] = set()
    removed_files = removed_bytes = 0

    def discard(path: Path) -> None:
        nonlocal removed_files, removed_bytes
        if path.is_dir() and not path.is_symlink():
            for current, _dirs, names in os.walk(path):
                for name in names:
                    removed_files += 1
                    removed_bytes += os.lstat(os.path.join(current, name)).st_size
            remove_tree(path)
        else:
            removed_files += 1
            removed_bytes += path.lstat().st_size
            path.unlink()

    def visit(directory: Path, selected: bool) -> None:
        for item in sorted(directory.iterdir()):
            relative = item.relative_to(root).as_posix()
            excluded = path_hits(relative, repo.path_exclude)
            matched.update(excluded)
            if excluded:
                discard(item)
                continue
            chosen = path_hits(relative, repo.path_only)
            matched.update(chosen)
            keep = selected or not repo.path_only or bool(chosen)
            if item.is_dir() and not item.is_symlink():
                if keep or could_hold_wanted(relative, repo.path_only):
                    visit(item, keep)
                    if not keep and not any(item.iterdir()):
                        item.rmdir()  # Nothing selected turned up inside it
                else:
                    discard(item)
            elif not keep:
                discard(item)

    visit(root, False)
    notes = [f"paths filter removed {removed_files:,} file(s), {removed_bytes / 2**20:,.0f} MiB"]
    notes += [f"paths pattern '{pattern}' matched nothing"
              for pattern in repo.path_only + repo.path_exclude if pattern not in matched]
    return notes


def force_remove(function: Any, path: str, _exc: BaseException) -> None:
    """rmtree's second chance at a path it could not remove.

    Removing an entry needs a writable, searchable PARENT, and descending into a directory needs
    rwx on the directory itself. A directory must therefore never be given a file's mode: 0600 on
    a directory takes away its search bit and makes everything beneath it unreachable.
    """
    os.chmod(os.path.dirname(path), stat.S_IRWXU)
    if not os.path.islink(path):
        os.chmod(path, stat.S_IRWXU if os.path.isdir(path) else stat.S_IRUSR | stat.S_IWUSR)
    function(path)


def remove_tree(path: Path) -> None:
    """Removes a tree even if it holds read-only entries, or is briefly still being written to
    (a git child that has been told to stop but has not quite gone)."""
    for attempt in range(4):
        try:
            shutil.rmtree(path, onexc=force_remove)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == 3:
                raise
            time.sleep(0.3 * (attempt + 1))
            for current, dirs, _files in os.walk(path):  # Reopen every directory, then retry
                for name in dirs + ["."]:
                    try:
                        os.chmod(os.path.join(current, name), stat.S_IRWXU)
                    except OSError:
                        pass


def strip_git_data(root: Path) -> int:
    """Removes every `.git` below root (the repo's own directory, and each submodule's file), and
    every upstream `.gitignore`, at any depth. Returns how many `.gitignore` files were removed.

    Ignore files keep working wherever they land, so inside this archive an upstream `.gitignore`
    would make `git add` silently skip files that upstream committed anyway (force-added build
    outputs, vendored blobs, fixtures). Whatever came down in the fetch is wanted here, so the
    ignore rules are dropped and the files they would have hidden are kept.
    """
    ignores = 0
    for current, dirs, files in os.walk(root):
        if ".git" in dirs:
            remove_tree(Path(current) / ".git")
            dirs.remove(".git")
        if ".git" in files:
            (Path(current) / ".git").unlink()
        if ".gitignore" in files:
            (Path(current) / ".gitignore").unlink()
            ignores += 1
    return ignores


ATTRIBUTES_LINE = "* -filter"


def ensure_filters_disabled(root: Path) -> None:
    """Makes sure the archive repository never runs a clean/smudge filter (LFS included).

    Snapshots keep their upstream `.gitattributes`, and those apply inside this repository too:
    a `filter=lfs` line would turn downloaded LFS content back into pointers at `git add`, and
    push the real bytes to the remote's LFS store. `$GIT_DIR/info/attributes` outranks every
    `.gitattributes` file, so one line there stores everything as ordinary blobs instead. The
    file is local to each clone and is never pushed, hence checking it on every run.
    """
    try:
        attributes = Path(git("rev-parse", "--git-path", "info/attributes", cwd=root))
    except GitError:
        print(f"Note: {root} is not inside a git repository, so no filter guard was written",
              file=sys.stderr)
        return
    if not attributes.is_absolute():
        attributes = root / attributes

    existing = attributes.read_text(encoding="utf-8") if attributes.exists() else ""
    if any(line.split() == ATTRIBUTES_LINE.split() for line in existing.splitlines()):
        return

    attributes.parent.mkdir(parents=True, exist_ok=True)
    with open(attributes, "a", encoding="utf-8") as file:
        if existing and not existing.endswith("\n"):
            file.write("\n")
        file.write("# Added by update.py: snapshots' own .gitattributes must never trigger filters\n"
                   "# (LFS especially), so that real content is committed as ordinary blobs.\n"
                   f"{ATTRIBUTES_LINE}\n")
    print(f"Wrote '{ATTRIBUTES_LINE}' to {attributes}")


def ensure_staging_excluded(root: Path) -> None:
    """Keeps the staging folder out of `git status`, via the clone-local `info/exclude`.

    A `.gitignore` in the root would do the same, but in single-repository mode the root's
    `.gitignore` belongs to upstream, so the rule lives where no snapshot can ever collide with it.
    """
    try:
        exclude = Path(git("rev-parse", "--git-path", "info/exclude", cwd=root))
    except GitError:
        return
    if not exclude.is_absolute():
        exclude = root / exclude
    existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    if f"{STAGING_PREFIX}*" in existing.splitlines():
        return
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with open(exclude, "a", encoding="utf-8") as file:
        file.write(("" if not existing or existing.endswith("\n") else "\n")
                   + f"# Added by update.py: only ever left behind if a run is killed\n{STAGING_PREFIX}*\n")


# --------------------------------------------------------------------------------------------
# Updating
# --------------------------------------------------------------------------------------------


@dataclass
class Outcome:
    repo: Repo
    status: str  # "updated" | "current" | "manual" | "failed"
    entry: dict[str, Any] | None = None
    detail: str = ""


PROTECTED: set[str] = {".git", MANIFEST_NAME}  # main() adds this script and the config by name


def swap_into_root(work: Path, root: Path, staging: Path,
                   previous_entries: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Single-repository mode: replaces the last snapshot's top-level entries with the new ones.

    The root is shared with this archive's own files, so nothing is touched that the manifest
    does not say came from the previous snapshot. An upstream entry is skipped if it would land on
    one of the archive's own files (PROTECTED) or on something that exists here and was not part
    of the last snapshot. The old entries are only destroyed once the new ones are in place, and
    are put back if anything fails halfway.
    """
    retired = Path(tempfile.mkdtemp(prefix="old-", dir=staging))
    moved_out: list[str] = []
    moved_in: list[str] = []
    skipped: dict[str, list[str]] = {"protected": [], "foreign": []}
    try:
        for entry in previous_entries:
            if entry not in PROTECTED and (root / entry).exists():
                (root / entry).rename(retired / entry)
                moved_out.append(entry)
        for item in sorted(work.iterdir(), key=lambda path: path.name):
            if item.name in PROTECTED or item.name.startswith(STAGING_PREFIX):
                skipped["protected"].append(item.name)
            elif (root / item.name).exists():
                skipped["foreign"].append(item.name)
            else:
                item.rename(root / item.name)
                moved_in.append(item.name)
    except OSError:
        for entry in moved_in:
            remove_tree(root / entry) if (root / entry).is_dir() else (root / entry).unlink()
        for entry in moved_out:
            (retired / entry).rename(root / entry)
        raise
    remove_tree(retired)
    remove_tree(work)
    return moved_in, skipped


def update_repo(repo: Repo, root: Path, staging: Path, previous: dict[str, Any] | None,
                refetch: bool) -> Outcome:
    if repo.manual:
        return Outcome(repo, "manual")

    target = root / repo.path
    name = repo.label
    work: Path | None = None
    try:
        STATUS.set(name, "checking remote")
        unchanged_config = previous is not None and all(
            previous.get(key) == value for key, value in repo.settings().items()
        )
        in_place = (all((root / entry).exists() for entry in (previous or {}).get("entries", [""]))
                    if repo.in_root else target.is_dir())
        # HIGHEST: an entry with no ref follows the highest version-named branch, chosen now.
        configured = repo
        resolved_ref = repo.ref
        tip: str | None = None
        fallback = False
        if repo.ref is None:
            found = highest_version_branch(repo.url)
            if found is not None:
                resolved_ref, tip = found
            else:
                resolved_ref, fallback = "HEAD", True
            repo = replace(repo, ref=resolved_ref)
        was_ref = (previous or {}).get("resolvedRef")

        if not refetch and unchanged_config and in_place and was_ref == resolved_ref:
            if (tip or remote_commit(repo)) == previous["commit"]:  # type: ignore[index]
                return Outcome(configured, "current", previous)

        work = Path(tempfile.mkdtemp(prefix="fetch-", dir=staging))
        work, commit, commit_date, notes = fetch_snapshot(repo, work)
        if configured.ref is None:
            if fallback:
                notes.insert(0, "no version-named branch; followed the default branch")
            elif was_ref and was_ref != resolved_ref:
                notes.insert(0, f"branch {was_ref} -> {resolved_ref}")
            else:
                notes.insert(0, f"branch {resolved_ref}")
        repo = configured  # The manifest records the CONFIGURED settings, plus what they resolved to

        if repo.in_root:
            STATUS.set(name, "swapping in")
            entries, skipped = swap_into_root(work, root, staging, (previous or {}).get("entries", []))
            notes += [f"kept this archive's own '{entry}' over upstream's" for entry in skipped["protected"]]
            notes += [f"left '{entry}' alone: it exists here and did not come from the snapshot"
                      for entry in skipped["foreign"]]
            entry_record = repo.settings() | {
                "commit": commit,
                "commitDate": commit_date,
                "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "entries": entries,
                "resolvedRef": resolved_ref,
            }
            before = previous["commit"][:10] if previous else "(new)"
            detail = f"{before} -> {commit[:10]}" + "".join(f"; {note}" for note in notes)
            return Outcome(repo, "updated", entry_record, detail)

        # Swap the new tree in; the old one is only destroyed once the new one is in place.
        STATUS.set(name, "swapping in")
        target.parent.mkdir(parents=True, exist_ok=True)
        retired = None
        if target.exists():
            retired = Path(tempfile.mkdtemp(prefix="old-", dir=staging)) / "tree"
            target.rename(retired)
        try:
            work.rename(target)
        except OSError:
            if retired is not None:
                retired.rename(target)
            raise
        if retired is not None:
            remove_tree(retired.parent)

        entry = repo.settings() | {
            "commit": commit,
            "commitDate": commit_date,
            "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "resolvedRef": resolved_ref,
        }
        before = previous["commit"][:10] if previous else "(new)"
        detail = f"{before} -> {commit[:10]}" + "".join(f"; {note}" for note in notes)
        return Outcome(repo, "updated", entry, detail)
    except (GitError, OSError) as error:
        # Clear the partial fetch away now; a half-cloned FreeCAD should not sit on the disk for
        # the rest of the run. Best effort: the end-of-run sweep of staging is the backstop.
        if work is not None and work.exists():
            STATUS.set(name, "cleaning up after failure")
            try:
                remove_tree(work)
            except OSError:
                pass
        return Outcome(repo, "failed", previous, str(error))
    finally:
        STATUS.done(name)


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    return manifest if isinstance(manifest, dict) else {}


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(dict(sorted(manifest.items())), file, indent=2)
        file.write("\n")

# --------------------------------------------------------------------------------------------
# README and setup
# --------------------------------------------------------------------------------------------


def render_readme(folder_name: str, config_name: str, remote_url: str | None,
                  single: bool = False) -> str:
    """The README text: prose wrapped at README_WIDTH, code blocks verbatim, no trailing spaces."""
    blocks: list[tuple[str, Any]] = [
        ("raw", f"# Shallow {folder_name} Repositories"),
        ("p", "This repository contains single-layer, full-breadth clones of key repositories for "
              f"archival purposes. It is handled using a tailored `update.py` script, which reads the list "
              f"of sources from `{config_name}` and replaces each snapshot with the newest state of "
              "its target branch."),
        ("p", "Each snapshot is the complete tree of its upstream at a single commit, with submodules "
              "and LFS content included and all git history removed. The code is kept here to be "
              "read, not to be built, so none of it is expected to compile in place."),
    ]
    if single:
        blocks = [
            ("raw", f"# Shallow {folder_name} Repository"),
            ("p", "This repository contains a single-layer, full-breadth clone of one upstream "
                  "repository for archival purposes, placed directly in the root. It is handled "
                  f"using the `update.py` script, which reads the source from `{config_name}` and "
                  "replaces the snapshot with the newest state of its target branch."),
            ("p", "The snapshot is the complete tree of its upstream at a single commit, with "
                  "submodules and LFS content included and all git history removed. The code is "
                  "kept here to be read, not to be built."),
            ("p", f"This file is `{ARCHIVE_README_NAME}` and not `README.md` because in this mode "
                  "`README.md` belongs to upstream. The archive's own files (`update.py`, "
                  f"`{config_name}`, `{MANIFEST_NAME}`, `{ARCHIVE_README_NAME}`) always win over an "
                  "upstream file of the same name, and anything else in the root that did not come "
                  "from the last snapshot is left alone; both cases are reported on the result "
                  "line."),
        ]
    if remote_url:
        blocks += [("raw", "## Cloning"), ("code", f"git clone {remote_url}")]
    blocks += [
        ("raw", "## Updating"),
        ("p", "The script needs python 3.12 or newer and `git` on the path, plus `git-lfs` if LFS "
              "content is wanted as real files and not pointers. Run it from the root of this "
              "repository, then commit and push the result:"),
        ("code", "./update.py\ngit add -A\ngit commit -m \"Update snapshots\"\ngit push"),
        ("p", "Any repository whose remote has not moved since the last run is skipped, so a re-run "
              "is cheap. A repository that fails does not stop the others; failures are listed at "
              "the end and the script exits with a failure status."),
    ]
    if single:
        blocks += [
            ("raw", "## Configuration"),
            ("p", f"`{config_name}` holds one object, which is the repository itself. Only `url` is "
                  "required:"),
            ("code", "{\n  \"url\": \"https://gitlab.com/kicad/code/kicad.git\",\n"
                     "  \"submodules\": { \"exclude\": [\"thirdparty/huge\"] },\n}"),
        ]
    else:
        blocks += [
            ("raw", "## Adding a Repository"),
            ("p", f"Edit `{config_name}`. An object is a folder, a list holds the repositories "
                  "that live in that folder, and the key `\".\"` means the folder itself, so the "
                  "top-level `\".\"` list holds repositories that land in the root. Only `url` is "
                  "required:"),
            ("code", "\"signals\": [\n  { \"url\": \"https://github.com/RustAudio/cpal\" },\n"
                     "  { \"url\": \"https://codeberg.org/someone/thing\", \"ref\": \"v1.2.0\" },\n],"),
        ]
    blocks += [
        ("ul", [
            "`url`: anything `git fetch` accepts, on any host.",
            *([] if single else ["`name`: the folder name for the snapshot. Defaults to the last "
                                 "segment of the URL."]),
            "`ref`: normally left out, in which case the snapshot follows the branch whose name is "
            "the highest version number (`0.4.3` over `0.4.2` over `0.4.2-rc.1`), chosen afresh on "
            "every run; a remote with no version-named branch falls back to its default branch. "
            "Otherwise a branch, tag, or full commit hash; `\"HEAD\"` means the default branch. A "
            "branch is followed to its newest tip on every run; a tag or hash stays where it is; "
            "`\"manual\"` skips the entry and prints a reminder.",
            "`submodules`: defaults to true, meaning all of them, recursively; false means none. To "
            "choose, give globs against the submodule paths: `{ \"exclude\": [\"src/llvm-project\"] }` "
            "fetches everything else, and `{ \"only\": [\"src/tools/*\"] }` fetches just those and "
            "whatever is nested inside them.",
            "`lfs`: defaults to true. Set it to false to keep LFS pointer files.",
            "`paths`: keeps only part of a snapshot, with the same `only` and `exclude` globs as "
            "`submodules`: `{ \"exclude\": [\"LayoutTests\", \"Websites\"] }`. The whole repository is "
            "still fetched, and the filter is applied before the snapshot is moved into place. A "
            "pattern selects a path and everything beneath it; with `only`, nothing else is kept.",
        ]),
        ("raw", "## Useful Flags"),
        ("ul", [
            "`--list`: show where every snapshot will land, without touching anything.",
            "`--only 'signals/*'`: only update snapshots whose path matches the glob. Repeatable.",
            "`--refetch`: fetch even when the remote has not moved.",
            "`--prune`: delete snapshots that are no longer in the config.",
            "`--no-submodules`, `--no-lfs`: override those settings for one run.",
            "`--jobs N`: the number of parallel fetches, 4 by default.",
            "`--create-config=\"foo,bar/baz\"`: write a fresh, commented config with those folders.",
            "`--create-single-config=URL`: write a config for single-repository mode, where one "
            "snapshot lands directly in the root.",
            "`--refresh-guide`: replace the guide comment at the top of the config with the current "
            "one, leaving the entries untouched.",
            "`--create-readme`: write this file.",
            "`--setup`: prepare a new archive repository: git, remote, README, and config.",
        ]),
        ("raw", "## Files"),
        ("ul", [
            f"`{config_name}`: the sources, edited by hand. Comments and trailing commas are fine.",
            f"`{MANIFEST_NAME}`: written by the script. It records the exact commit each snapshot was "
            "taken at. It is a record and never a pin, and does not need editing.",
            "`update.py`: the script itself.",
        ]),
        ("raw", "## Notes"),
        ("ul", [
            "Nothing is ever pushed as LFS. The script keeps the line `* -filter` in this "
            "repository's `info/attributes`, so LFS content is committed as ordinary files. That "
            "file is local to each clone, so run `update.py` at least once in a fresh clone before "
            "committing snapshots from it.",
            "Line endings follow each upstream's own `.gitattributes`, which travel with the "
            "snapshot.",
            "Upstream `.gitignore` files are removed from every snapshot, at every depth. Left in "
            "place they would make `git add` skip files that upstream committed anyway; "
            "everything that comes down in a fetch is meant to be kept here.",
        ]),
    ]

    parts: list[str] = []
    for kind, content in blocks:
        if kind == "p":
            parts.append(textwrap.fill(content, width=README_WIDTH, break_long_words=False,
                                       break_on_hyphens=False))
        elif kind == "ul":
            parts.append("\n".join(
                textwrap.fill(item, width=README_WIDTH, initial_indent="- ", subsequent_indent="  ",
                              break_long_words=False, break_on_hyphens=False)
                for item in content))
        elif kind == "code":
            parts.append(f"```\n{content}\n```")
        else:
            parts.append(content)
    lines = [line.rstrip() for line in "\n\n".join(parts).splitlines()]
    return "\n".join(lines) + "\n"


def origin_url(root: Path) -> str | None:
    try:
        return git("remote", "get-url", "origin", cwd=root) or None
    except GitError:
        return None


def create_readme(root: Path, config_name: str, remote_url: str | None,
                  single: bool = False) -> None:
    readme = root / (ARCHIVE_README_NAME if single else README_NAME)
    if readme.exists():
        raise ConfigError(f"{readme} already exists; refusing to overwrite it")
    root.mkdir(parents=True, exist_ok=True)
    readme.write_text(render_readme(root.name, config_name, remote_url or origin_url(root), single),
                      encoding="utf-8")
    print(f"Wrote {readme}")


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def run_setup(root: Path, config_path: Path, remote_url: str | None, folders: str | None,
              single: bool = False, source_url: str | None = None) -> None:
    """Prepares a new archive repository. Anything passed on the command line is not asked for."""
    root.mkdir(parents=True, exist_ok=True)

    # 1. The remote
    if remote_url is None:
        remote_url = ask("Remote URL (blank for none): ")
    try:
        git("rev-parse", "--git-dir", cwd=root)
    except GitError:
        git("init", "--quiet", str(root))
        print(f"Initialised a git repository in {root}")
    existing = origin_url(root)
    if remote_url and existing is None:
        git("remote", "add", "origin", remote_url, cwd=root)
        print(f"Set origin to {remote_url}")
    elif remote_url and existing != remote_url:
        print(f"Note: origin is already {existing}; left as it is")
    ensure_filters_disabled(root)
    ensure_staging_excluded(root)

    # 2. The README
    try:
        create_readme(root, config_path.name, remote_url or None, single)
    except ConfigError as error:
        print(f"Note: {error}")

    # 3. The config
    if single:
        if source_url is None:
            source_url = ask("URL of the repository to snapshot into the root: ")
        try:
            create_single_config(config_path, source_url)
        except ConfigError as error:
            print(f"Note: {error}")
        return
    if folders is None:
        print("Folder names, one per line (nest with '/'); a blank line finishes:")
        entered: list[str] = []
        while line := ask("  folder: "):
            try:
                for segment in line.strip("/").split("/"):
                    check_segment(segment, f"'{line}'")
                entered.append(line)
            except ConfigError as error:
                print(f"  {error}; try again")
        folders = ",".join(entered)
    try:
        create_config(config_path, [folders])
    except ConfigError as error:
        print(f"Note: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="update.py",
        description="Updates the code snapshots in this repository from the sources in the "
                    "configuration file. Requires python >= 3.12 and `git` on the system path.",
    )
    parser.add_argument("--config", default=None,
                        help="path to the configuration file, JSON or JSONC "
                             "(default: ./config.jsonc, else ./config.json)")
    parser.add_argument("--root", default=None,
                        help="where snapshots land (default: the folder holding the config)")
    parser.add_argument("--only", action="append", default=[], metavar="GLOB",
                        help="only touch snapshots whose path matches, e.g. 'signals/*' or "
                             "'*/serde'; repeatable")
    parser.add_argument("--refetch", action="store_true",
                        help="fetch even when the remote has not moved since the manifest")
    parser.add_argument("--no-submodules", action="store_true",
                        help="skip submodules for every repository in this run")
    parser.add_argument("--no-lfs", action="store_true",
                        help="leave LFS files as pointers for every repository in this run")
    parser.add_argument("--jobs", type=int, default=4, help="parallel fetches (default 4)")
    parser.add_argument("--list", action="store_true",
                        help="print what the config resolves to and exit; touches nothing")
    parser.add_argument("--create-config", action="append", default=[], metavar="FOLDERS",
                        help="write a scratch config (at --config) with empty lists for the "
                             "given comma-separated folders, e.g. 'foo,bar,baz/quux', then exit; "
                             "never overwrites an existing file")
    parser.add_argument("--refresh-guide", action="store_true",
                        help="replace the guide comment at the top of the existing config with "
                             "this version's; the config body is left untouched")
    parser.add_argument("--create-readme", action="store_true",
                        help=f"write a basic {README_NAME} for this archive, then exit; never "
                             "overwrites an existing file")
    parser.add_argument("--setup", action="store_true",
                        help="prepare a new archive here: git init, origin, the LFS guard, "
                             ".gitignore, README, and config. Asks for the remote URL and the "
                             "folder names unless --remote-url / --folders are given")
    parser.add_argument("--remote-url", default=None, metavar="URL",
                        help="with --setup or --create-readme: the remote ('' for none)")
    parser.add_argument("--folders", default=None, metavar="FOLDERS",
                        help="with --setup: comma-separated folders, as for --create-config "
                             "('' for none)")
    parser.add_argument("--create-single-config", default=None, metavar="URL",
                        help="write a config for single-repository mode: one snapshot, whose "
                             "contents land directly in the root; then exit")
    parser.add_argument("--single", action="store_true",
                        help="with --setup or --create-readme: single-repository mode (setup asks "
                             "for the source URL instead of folders; the readme is ARCHIVE.md)")
    parser.add_argument("--source-url", default=None, metavar="URL",
                        help="with --setup --single: the repository to snapshot, so it is not asked")
    parser.add_argument("--prune", action="store_true",
                        help="delete snapshots that are in the manifest but no longer in the config")
    args = parser.parse_args()

    if sys.version_info < (3, 12):
        raise SystemExit("This script requires python >= 3.12")
    if not shutil.which("git"):
        raise SystemExit("This script requires that the `git` command be on the system path")

    if args.config:
        config_path = Path(args.config).resolve()
    else:
        present = [Path(name).resolve() for name in DEFAULT_CONFIG_NAMES if Path(name).exists()]
        if len(present) > 1 and not (args.create_config or args.setup or args.create_readme
                                     or args.create_single_config):
            raise SystemExit("Both config.jsonc and config.json exist here; pass --config to choose")
        config_path = present[0] if present else Path(DEFAULT_CONFIG_NAMES[0]).resolve()
    root = Path(args.root).resolve() if args.root else config_path.parent

    if args.setup:
        run_setup(root, config_path, args.remote_url, args.folders,
                  args.single or args.source_url is not None, args.source_url)
        return 0
    if args.create_readme or args.create_config or args.refresh_guide or args.create_single_config:
        try:
            if args.refresh_guide:
                refresh_config_guide(config_path)
            if args.create_single_config:
                create_single_config(config_path, args.create_single_config)
            if args.create_readme:
                single = args.single or bool(args.create_single_config) or (
                    config_path.exists() and any(r.in_root for r in load_config(config_path)))
                create_readme(root, config_path.name, args.remote_url, single)
            if args.create_config:
                create_config(config_path, args.create_config)
        except (ConfigError, json.JSONDecodeError) as error:
            raise SystemExit(f"Config error: {error}")
        return 0
    try:
        repos = load_config(config_path)
    except (ConfigError, json.JSONDecodeError) as error:
        raise SystemExit(f"Config error: {error}")

    if args.no_submodules:
        repos = [replace(r, submodules=False, sub_only=(), sub_exclude=()) for r in repos]
    if args.no_lfs:
        repos = [replace(r, lfs=False) for r in repos]

    manifest_path = root / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    PROTECTED.update({Path(__file__).name, config_path.name, *DEFAULT_CONFIG_NAMES, ARCHIVE_README_NAME})

    if args.list:
        for repo in repos:
            setting = repo.submodule_setting()
            flags = "" if setting is True else (
                "  [no submodules]" if setting is False else f"  [submodules: {json.dumps(setting)}]")
            if repo.path_setting():
                flags += f"  [paths: {json.dumps(repo.path_setting())}]"
            where = "(root)  single-repository mode" if repo.in_root else str(repo.path)
            print(f"{where:<40} {repo.ref or '(highest)':<12} {repo.url}{flags}")
        return 0

    selected = [r for r in repos
                if not args.only or any(fnmatch.fnmatch(str(r.path), g) for g in args.only)]
    if not selected:
        raise SystemExit("Nothing in the config matches the given --only patterns")

    if any(r.lfs for r in selected) and subprocess.run(
            ["git", "lfs", "version"], capture_output=True).returncode != 0:
        print("Note: git-lfs is not installed, so LFS-tracked files will arrive as pointer files",
              file=sys.stderr)

    # Staging lives inside the root so that the final renames never cross a filesystem.
    root.mkdir(parents=True, exist_ok=True)
    ensure_filters_disabled(root)
    ensure_staging_excluded(root)
    staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=root))
    outcomes: list[Outcome] = []
    interrupted = False
    try:
        STATUS.total = len(selected)

        def handle_outcome(count: int, outcome: Outcome) -> None:
            outcomes.append(outcome)
            STATUS.finished = count
            if outcome.status == "updated" and outcome.entry is not None:
                manifest[str(outcome.repo.path)] = outcome.entry
            STATUS.emit(f"[{count}/{len(selected)}] {outcome.status:<8} {outcome.repo.label}"
                        f"{'  ' + outcome.detail if outcome.detail else ''}")

        with STATUS, ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = [pool.submit(update_repo, repo, root, staging,
                                   manifest.get(str(repo.path)), args.refetch)
                       for repo in selected]
            try:
                completed = enumerate(as_completed(futures), start=1)
                for count, future in completed:
                    handle_outcome(count, future.result())
            except KeyboardInterrupt:
                # Without this, leaving the pool would first run every queued snapshot to completion.
                ABORT.set()
                interrupted = True
                for future in futures:
                    future.cancel()
                stop_running_git()
                STATUS.emit("Interrupted: stopping the running fetches and cleaning up...")
    finally:
        write_manifest(manifest_path, manifest)
        try:
            remove_tree(staging)
        except OSError as error:
            print(f"Note: could not remove {staging} ({error}). It is safe to delete by hand:\n"
                  f"  chmod -R u+rwx '{staging}'; rm -rf '{staging}'", file=sys.stderr)
    if interrupted:
        return 130

    # Snapshots the config no longer mentions
    configured = {str(repo.path) for repo in repos}
    for orphan in sorted(set(manifest) - configured):
        if orphan == ".":  # A former single-repository snapshot: never sweep the root itself
            print("Note: the manifest holds a single-repository snapshot the config no longer has; "
                  "its files were left in place")
            continue
        if args.prune:
            if (root / orphan).is_dir():
                remove_tree(root / orphan)
            del manifest[orphan]
            print(f"Pruned {orphan}")
        else:
            print(f"Note: {orphan} is in the manifest but not the config (--prune removes it)")
    if args.prune:
        write_manifest(manifest_path, manifest)

    for outcome in outcomes:
        if outcome.status == "manual":
            print(f"Repository {outcome.repo.label} needs to be filled manually from "
                  f"{outcome.repo.url}")

    failed = [o for o in outcomes if o.status == "failed"]
    for outcome in failed:
        print(f"FAILED {outcome.repo.label}: {outcome.detail}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
