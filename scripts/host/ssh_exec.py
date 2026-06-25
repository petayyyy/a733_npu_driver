#!/usr/bin/env python3
"""Small password-based SSH/SFTP helper for A733 board bring-up."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import posixpath
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYDEPS = REPO_ROOT / "work" / "pydeps"
if PYDEPS.exists():
    sys.path.insert(0, str(PYDEPS))

try:
    import paramiko
except ImportError as exc:  # pragma: no cover - exercised only on missing deps.
    raise SystemExit(
        "Missing paramiko. Run: python -m pip install --target work\\pydeps paramiko"
    ) from exc


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    password = os.environ.get(args.password_env)
    if not password:
        raise SystemExit(f"Environment variable {args.password_env} is not set")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=args.host,
        port=args.port,
        username=args.user,
        password=password,
        timeout=args.timeout,
        banner_timeout=args.timeout,
        auth_timeout=args.timeout,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def run_command(client: paramiko.SSHClient, command: str, timeout: int, get_pty: bool) -> int:
    print(f"[remote] {command}", flush=True)
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout, get_pty=get_pty)
    stdin.close()

    for line in stdout:
        try:
            print(line, end="")
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"), end="")
    for line in stderr:
        try:
            print(line, end="", file=sys.stderr)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"), end="", file=sys.stderr)

    status = stdout.channel.recv_exit_status()
    print(f"[remote-exit] {status}", flush=True)
    return status


def mkdir_p(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = [part for part in remote_dir.split("/") if part]
    current = "/" if remote_dir.startswith("/") else "."
    for part in parts:
        current = posixpath.join(current, part)
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def upload_path(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    if local.is_dir():
        mkdir_p(sftp, remote)
        for child in local.iterdir():
            upload_path(sftp, child, posixpath.join(remote, child.name))
        return

    remote_dir = posixpath.dirname(remote)
    if remote_dir:
        mkdir_p(sftp, remote_dir)
    print(f"[put] {local} -> {remote}", flush=True)
    sftp.put(str(local), remote)


def download_path(sftp: paramiko.SFTPClient, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get] {remote} -> {local}", flush=True)
    sftp.get(remote, str(local))


def parse_pair(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("expected LOCAL:REMOTE or REMOTE:LOCAL")
    left, right = value.split(":", 1)
    if not left or not right:
        raise argparse.ArgumentTypeError("both sides of pair must be non-empty")
    return left, right


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="radxa")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--password-env", default="A733_SSH_PASSWORD")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--pty", action="store_true", help="request a pseudo-terminal")
    parser.add_argument("--run", action="append", default=[], help="remote command to run")
    parser.add_argument("--put", action="append", type=parse_pair, default=[], help="upload LOCAL:REMOTE")
    parser.add_argument("--get", action="append", type=parse_pair, default=[], help="download REMOTE:LOCAL before commands")
    parser.add_argument(
        "--get-after",
        action="append",
        type=parse_pair,
        default=[],
        help="download REMOTE:LOCAL after commands",
    )
    args = parser.parse_args()

    client = connect(args)
    try:
        if args.put or args.get:
            with client.open_sftp() as sftp:
                for local, remote in args.put:
                    upload_path(sftp, Path(local), remote)
                for remote, local in args.get:
                    download_path(sftp, remote, Path(local))

        status = 0
        for command in args.run:
            status = run_command(client, command, args.timeout, args.pty)
            if status != 0:
                return status
        if args.get_after:
            with client.open_sftp() as sftp:
                for remote, local in args.get_after:
                    download_path(sftp, remote, Path(local))
        return status
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
