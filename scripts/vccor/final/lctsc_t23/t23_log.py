"""vccor/final LCTSC-T23 registration-log helpers (DERIVED from the frozen lctsc_log.py by derive_t23.py): lines appended
to the project's final_prereg.log in its existing formats ('[time] KIND sha256 <h> <rel path> (note)' and
'[time] KIND lctsc_t23 <text>'), plus a plain event log. Standard library only.
"""

import hashlib
import json
import os
import time
from pathlib import Path

REPO = Path("/home/user/Documents/Medical Imagining SSL")
PREREG_LOG = REPO / "outputs/logs/vccor/final_prereg.log"
EVENTS = REPO / "outputs/logs/vccor/final_lctsc_t23.log"
TAG = "lctsc_t23"
if os.environ.get("LCTSC_T23_REHEARSAL_ROOT"):
    PREREG_LOG = Path(os.environ["LCTSC_T23_REHEARSAL_ROOT"]) / "prereg.log"
    EVENTS = Path(os.environ["LCTSC_T23_REHEARSAL_ROOT"]) / "events.log"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def _append(path, line):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def prereg_line(kind, path, note):
    h = sha256(path)
    _append(PREREG_LOG, f"[{now()}] {kind} sha256 {h} {rel(path)} ({note})")
    return h


def prereg_note(kind, text):
    _append(PREREG_LOG, f"[{now()}] {kind} {TAG} {text}")


def event(text):
    _append(EVENTS, f"[{now()}] {text}")


def write_json(path, obj, readonly=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)
    if readonly:
        os.chmod(path, 0o444)
    return sha256(path)
