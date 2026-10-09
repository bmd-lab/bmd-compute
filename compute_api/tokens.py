"""Generate a machine-API token and the matching token-file entry.

Usage::

    python -m compute_api.tokens --principal NAME --scope plan [--scope read]

The token is printed once on standard output and is not stored anywhere. Add
the printed entry to the deployment-local file named by ``BMD_API_TOKENS_FILE``
(outside the repository, readable only by the service account), and give the
token to the client through a private channel.
"""

from __future__ import annotations

import argparse
import json
import sys

from compute_api.auth import (
    KNOWN_SCOPES,
    PRINCIPAL_PATTERN,
    TOKENS_FILE_SCHEMA,
    TOKENS_FILE_SCHEMA_VERSION,
    generate_token,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m compute_api.tokens")
    parser.add_argument("--principal", required=True)
    parser.add_argument("--scope", action="append", required=True, choices=sorted(KNOWN_SCOPES))
    args = parser.parse_args(argv)
    if not PRINCIPAL_PATTERN.fullmatch(args.principal):
        parser.error("principal must match ^[a-z0-9][a-z0-9._-]{0,63}$")

    token, token_id, verifier = generate_token()
    entry = {
        "principal": args.principal,
        "token_id": token_id,
        "verifier": verifier,
        "scopes": sorted(set(args.scope)),
        "enabled": True,
    }
    sys.stdout.write("Token (shown once; store it privately on the client):\n")
    sys.stdout.write(f"{token}\n\n")
    sys.stdout.write(
        "Add this entry to the 'principals' list of the server token file "
        f"(schema {TOKENS_FILE_SCHEMA} v{TOKENS_FILE_SCHEMA_VERSION}):\n"
    )
    sys.stdout.write(json.dumps(entry, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
