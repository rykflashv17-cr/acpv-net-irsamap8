#!/usr/bin/env python3
"""Read-only, pre-write storage check. This is not a filesystem hard quota."""
import argparse
import json
import os
from pathlib import Path
import stat
import sys


def usage(root):
    logical = allocated = files = transient = 0
    pending = [root]
    seen = set()
    while pending:
        path = pending.pop()
        try:
            info = path.lstat()
        except FileNotFoundError:
            if path == root:
                raise
            # Atomic .tmp -> final renames can race with a read-only tree scan.
            # Charge a generous reserve per vanished entry instead of ignoring it.
            transient += 1
            continue
        key = (info.st_dev, info.st_ino)
        if key in seen:
            continue
        seen.add(key)
        logical += info.st_size
        allocated += info.st_blocks * 512
        if stat.S_ISDIR(info.st_mode):
            try:
                with os.scandir(path) as entries:
                    pending.extend(Path(entry.path) for entry in entries)
            except FileNotFoundError:
                transient += 1
        elif stat.S_ISREG(info.st_mode):
            files += 1
    reserve = transient * 64 * 1024**2
    return {"logical_bytes": logical, "allocated_bytes": allocated,
            "effective_bytes": max(logical, allocated) + reserve, "files": files,
            "scan_transient_entries": transient, "scan_transient_reserve_bytes": reserve}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pending-bytes", type=int, default=0,
                        help="Conservative maximum additional bytes for the next bounded batch.")
    args = parser.parse_args()
    if args.pending_bytes < 0:
        parser.error("--pending-bytes must be nonnegative")
    root = Path(os.environ.get('ACPV_WORKSPACE',str(Path(__file__).resolve().parent))).resolve()
    plan = json.loads((root / "storage_budget.json").read_text())
    used = usage(root)
    fs = os.statvfs(root)
    available = fs.f_bavail * fs.f_frsize
    projected = used["effective_bytes"] + args.pending_bytes
    reasons = []
    if projected >= plan["stop_new_writes_at_bytes"]:
        reasons.append("Projected project usage reaches the configured stop threshold.")
    if available - args.pending_bytes < plan["filesystem_minimum_available_bytes"]:
        reasons.append("Shared filesystem available space would fall below the configured reserve.")
    result = {"workspace": str(root), "usage": used,
              "pending_bytes": args.pending_bytes, "projected_bytes": projected,
              "stop_threshold_bytes": plan["stop_new_writes_at_bytes"],
              "filesystem_available_bytes": available,
              "quota_verified": plan["personal_or_fileset_quota_verified"],
              "filesystem_hard_quota_enforced": False,
              "guard_status": "STOP" if reasons else "PASS", "reasons": reasons}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if reasons else 0


if __name__ == "__main__":
    sys.exit(main())
