"""
Runs a command and kills it (with all its child processes) if it uses too much RAM.

Memory is measured every few seconds as the sum of PSS over the whole process tree, so worker processes
are included and memory shared between forked workers is not counted twice.
The job is also killed if the free memory of the server drops below --min_free_gb (protects other jobs).

Example (from the repository root):
    python memlimit.py --limit_gb 150 -- python node2vec_baseline.py --inference random_link_split --seed 1

Exit code: the exit code of the command, or 137 if it was killed.
The memory log is written to memlimit.log (or --log).
"""

import os
import sys
import time
import signal
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--limit_gb", required=False, default=150, type=float, help="kill the job above this RAM use")
parser.add_argument("--min_free_gb", required=False, default=50, type=float, help="kill the job if server free RAM drops below this")
parser.add_argument("--interval", required=False, default=5, type=float, help="seconds between checks")
parser.add_argument("--log", required=False, default="memlimit.log", help="memory log file")
parser.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to run")


# all processes started by the job (children, grandchildren, ...)
def process_tree(root):
    children = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            children.setdefault(ppid, []).append(int(d))
        except (OSError, IndexError, ValueError):
            pass
    tree, stack = [], [root]
    while stack:
        p = stack.pop()
        tree.append(p)
        stack.extend(children.get(p, []))
    return tree


def pss_gb(pid):
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return 0.0


def server_free_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    return float("inf")


def main():
    args = parser.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        parser.error("no command given, use: python memlimit.py --limit_gb 150 -- python script.py ...")

    log = open(args.log, "a", buffering=1)
    start = time.time()
    proc = subprocess.Popen(cmd, start_new_session=True)  # own process group, so all workers can be killed together
    log.write(f"started pid {proc.pid}: {' '.join(cmd)}\n")

    peak, last_log, killed = 0.0, -60.0, False
    while proc.poll() is None:
        used = sum(pss_gb(p) for p in process_tree(proc.pid))
        free = server_free_gb()
        peak = max(peak, used)
        minutes = (time.time() - start) / 60

        if minutes * 60 - last_log >= 60:
            log.write(f"[{minutes:7.1f} min] used {used:7.1f} GB | peak {peak:7.1f} GB | server free {free:6.1f} GB\n")
            last_log = minutes * 60

        if used > args.limit_gb or free < args.min_free_gb:
            reason = (f"used {used:.1f} GB > limit {args.limit_gb} GB" if used > args.limit_gb
                      else f"server free {free:.1f} GB < {args.min_free_gb} GB")
            msg = f"[{minutes:7.1f} min] KILLED: {reason}"
            log.write(msg + "\n")
            print(msg, file=sys.stderr)
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            killed = True
            break
        time.sleep(args.interval)

    status = "killed" if killed else f"exit code {proc.returncode}"
    log.write(f"finished: {status} | peak {peak:.1f} GB | {(time.time() - start) / 60:.1f} min\n")
    sys.exit(137 if killed else proc.returncode)


if __name__ == "__main__":
    main()
