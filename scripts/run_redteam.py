#!/usr/bin/env python
"""Thin wrapper around agentic_redteam.cli.run_redteam_main."""

from agentic_redteam.cli import run_redteam_main

if __name__ == "__main__":
    raise SystemExit(run_redteam_main())
