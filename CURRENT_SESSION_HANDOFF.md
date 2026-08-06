# Current Session Handoff

Date: 2026-07-19

## Purpose

This is a concise capture of the current working session for transfer with the
rover workspace. It is not a verbatim ChatGPT transcript. The older detailed
conversation records remain available in:

- `chat_handoff.md`
- `chat_session_reconstructed.md`
- `codex_handoff_prompt.md`
- `CONTEXT.md`

## Documentation And Workspace Consolidation

- `live_indoor_runtime.py` is the maintained indoor/no-GPS live runtime.
- The superseded MBRA-first and heavy-recovery variants, plus the obsolete
  `mbra_repo_1` controller adapter, are retained under `legacy/` for
  provenance only.
- The no-GPS operating brief is
  `obsidian_vault/00 Home/Current No-GPS - Read This First.md`.
- Field truth is
  `obsidian_vault/01 Source of Truth/No-GPS Field Trial - Findings.md`.
- `obsidian_vault/` is its own Git repository with a separate remote.
- `mbra_repo_1/` and `nyu-earthrover-main/` are reference workspaces, not the
  current route-repeat runtime path.

## External Samsung SSD

- The external drive was initially detected only as the Ugreen Realtek RTL9210
  enclosure and reported zero capacity.
- After reseating, it is recognized as `/dev/sda`, model `SSD_990_EVO_Plus`,
  with 1 TB decimal capacity (`931.5 GiB` shown by Linux).
- It currently has no partition table, partition, filesystem, label, or mount
  point. It has not been formatted in this session.
- Recommended setup:
  - `GPT + ext4` for Linux-only rover logs, datasets, and model files.
  - `GPT + exFAT` if the SSD will regularly move between Linux, Windows, and
    macOS systems.

## Transfer Archive

The requested workspace archive includes hidden files, Git histories, local
configuration, recordings, model weights, and this handoff. It therefore may
contain secrets such as SDK credentials and Git configuration; keep it private.
