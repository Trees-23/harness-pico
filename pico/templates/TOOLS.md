# Tool Usage Notes

Tool signatures are provided automatically via function calling. These tools are model-callable
capabilities, not REPL commands for the user to type directly.

If the user lists tool names or asks what tools do, explain the tools in plain language.
Do not call tools just because a tool name appears in the user's message; call a tool only when
the user's actual task requires reading, searching, editing, scheduling, or running something.

`multi_tool_use.parallel` is not a user-facing Pico tool. Do not present it as something the user
can invoke from `pico>`.

## list_dir - File Discovery

- Use `list_dir` to inspect directories inside the workspace.
- Prefer this over `exec` when you only need file paths.

## grep and glob - Content Search

- Use `grep` to search file contents inside the workspace.
- Use `glob` to find files by path pattern.
- Prefer these over `exec` for code and history searches.
- Read only the relevant files or ranges after search narrows the scope.

## read_file - File Reading

- Use `read_file` for UTF-8 text files when exact content is needed.
- Avoid repeated overlapping reads; reuse previous results when possible.

## write_file and edit_file - File Editing

- Use `edit_file` for targeted replacements in existing files.
- Use `write_file` when creating a file or replacing an entire file is appropriate.
- Editing tools may require approval depending on policy.

## exec - Command Execution

- Use `exec` for tests, scripts, and commands that cannot be handled by file/search tools.
- Commands run in the workspace and may require approval depending on policy.
- Prefer built-in file/search tools for simple inspection.

## cron — Scheduled Reminders

- Use `cron` to list, add, or remove scheduled Pico jobs.
- Do not simulate reminders by only writing memory files.

## delegate and spawn - Child Investigation

- Use these for bounded read-only investigation tasks when available.
- Do not use child agents for simple questions that can be answered directly.

## notebook_edit - Notebook Editing

- Use `notebook_edit` to insert, replace, or delete Jupyter notebook cells.
