# Local product commands

Reference for [Kilix Ollama](../README.md).

Open the explicit development TUI in a real terminal. It contacts no runtime,
downloads no model, and treats pasted input as inert until the external Kilix
pane integration and prompt editor are accepted:

```sh
./kilix-ollama tui
```

The local TUI owns alternate-screen and terminal-mode cleanup. The separate
real-Kilix nested-pane and hard-crash force-restore proof remains an external
P9 gate.

Browse the exact three-entry metadata catalog without contacting a runtime, or
merge installed state from the private mediation socket:

```sh
./kilix-ollama library
./kilix-ollama library --installed
./kilix-ollama models
./kilix-ollama ps
./kilix-ollama show granite4.1:3b --development-runtime
```

Run one prompt from arguments, standard input, or one non-symlinked UTF-8 file.
Plain output is sanitized while streaming; JSON, JSON Lines, and atomic file
output are also available:

```sh
./kilix-ollama ask granite4.1:3b --development-runtime 'Explain this tree.'
./kilix-ollama ask granite4.1:3b --development-runtime --file prompt.txt --json
./kilix-ollama ask granite4.1:3b --development-runtime --output answer.md 'Summarize.'
```

Start a persistent terminal chat, run a single scripted turn, or manage the
owner-private conversation store:

```sh
./kilix-ollama chat granite4.1:3b --development-runtime
./kilix-ollama chat granite4.1:3b --development-runtime --message 'Hello' --json
./kilix-ollama chat granite3.2-vision:2b --development-runtime --message 'Describe this.' --attach image.png --json
./kilix-ollama chat --conversation CONVERSATION_ID --regenerate --json --development-runtime
./kilix-ollama chat --conversation CONVERSATION_ID --switch-model granite3.2-vision:2b --message 'Continue' --json --development-runtime
./kilix-ollama conversations list
./kilix-ollama conversations fork CONVERSATION_ID --messages 2 --title 'Alternative'
./kilix-ollama conversations export CONVERSATION_ID --output transcript.md
```

Completed turns commit as one compare-and-swap update. Generation cancelled
before commit, malformed tool output, an exhausted tool budget, or a concurrent
conversation update leaves the previously stored conversation intact.
Regeneration replaces only the suffix after the last user message. Forking
refuses a prefix that splits a tool exchange, and model switching verifies the
target is installed before it changes stored state. Interactive chat also
accepts `/regenerate`, `/fork`, and `/model MODEL`.

`--attach` accepts up to four explicitly selected PNG/JPEG files for a scripted
`--message`. The client opens each source-path component through no-follow
directory descriptors, enforces per-file and aggregate byte limits plus bounded
image dimensions/pixels, and retains the validated bytes in memory until the
turn succeeds. Before first use, explicitly provision the owner-only snapshot
directory:

```sh
install -d -m 0700 "${XDG_DATA_HOME:-$HOME/.local/share}/kilix-ollama/attachments"
```

The client never creates or repairs that trust root. It copies successful turns
through its verified descriptor into mode-0600 content-addressed snapshots
before committing the conversation. Invalid input, unavailable or failed
inference, a stale conversation race, or a missing/unsafe trust root publishes
no new snapshot. Conversation records retain only typed digest, media, size and
dimension metadata—never the selected path or filename. Every reuse rechecks
the snapshot digest and image metadata before runtime contact.
Markdown export states the omitted image count and never embeds attachment
bytes implicitly.

`--tools` offers exactly four read-only tools: UTC time, bounded arithmetic,
reading an explicitly selected text file, and bounded search beneath an
explicitly selected directory. Every call still requires interactive approval.
Tools are absent from the request by default. `tools --json` prints the exact
registry without enabling it.

No prompt command performs an implicit pull. `pull` currently fails before
runtime or network access because the shared F100 carrier/receipt authority is
0 of 1 available. `--development-runtime` admits only the exact local R3
candidate; it never admits an unknown runtime, cloud endpoint, or cloud-tagged
model.

Service descriptions use only the fixed root-owned launcher
`/usr/lib/plebian-os/ollama/ollama-current`. Diagnostics reject a symlink,
unexpected owner, group/world-writable component, or substituted ancestor in
that trusted path.

Plebian OS carries the audited private-listener patch and a small fail-closed
runtime installer. The verified native closure is not shipped, and no public
carrier has been accepted. Development installation therefore requires the
exact local closure explicitly:

```sh
sudo plebian-os-install-deps --vulkan-nouveau
sudo plebian-os-install-kilix-ollama-runtime --closure /path/to/exact/closure
```

The installer verifies the complete 119,645,888-byte executable/library set,
relative library links, notices, runtime package tuple, and downstream patch
identity before publishing the fixed launcher atomically. It downloads
nothing and contains no model or weight artifact.
