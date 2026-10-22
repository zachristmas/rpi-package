# rpi-herdr

Makes a pane running interactive `rpi` show up in [Herdr](https://github.com/herdrdev/herdr) as an agent, using Herdr's documented "Add Herdr support to your agent" contract. Herdr then shows `rpi` and its `idle`, `working` or `blocked` state in the sidebar and in `herdr agent list`, notifies on completion, supports `herdr agent wait`, and reopens the same session after a Herdr server restart.

Outside Herdr (no `HERDR_ENV=1`, `HERDR_PANE_ID` or `HERDR_BIN_PATH`) the extension registers nothing. Headless runs (`rpi -p`, RPC, piped stdio) never report.

## What it reports

| rpi event | Herdr state |
| --- | --- |
| interactive TUI start, session start or switch | `idle` (with the resume command) |
| agent run starts | `working` |
| a selector or dialog opens during a run (ask-user, permission gate) | `blocked` |
| agent run ends | `idle` |
| session shutdown | release |

A selector opened while the agent is idle (for example `/model`) does not change the state. Reports run on a background thread with a short timeout and failures are ignored, so Herdr can never slow rpi down. Each report carries an increasing sequence number so late reports cannot overwrite newer state.

The resume command is `rpi --session <id>`, with the id taken from `RPI_SESSION_ID`.

## Install

```sh
rpi install --path packages/rpi-herdr --force
```

Check from a Herdr pane running `rpi`: `herdr agent list`.

## Limits

Herdr's `agent start --kind` only knows built-in agent kinds, so another agent cannot launch rpi that way. Start `rpi` in a pane yourself; the pane then answers to `herdr agent get|read|prompt|wait <pane-id>`.
