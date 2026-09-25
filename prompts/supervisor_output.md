<!-- Appended after the supervisor observation. Placeholders: {title}, {repo_id}, {key}. -->
FINAL RESPONSE PROTOCOL: Return only the JSON action and instruction. Choose ONE bounded work unit,
not an entire roadmap. Routine short gates on frozen source typically batch up to four sequential
gates or roughly six minutes; stop a batch early on an unexpected failure, source drift or blockage,
preserving pending handles and stage dependencies. A single longer unit (up to ~25 minutes) is
acceptable when the worker reports clearly promising in-flight work — require a STATE.md progress
checkpoint before it starts. Do not require a new worker turn after every successful routine gate.
Target 200-700 characters for instruction; the hard limit is 2000. Do not repeat TASK.md, the full
gate matrix, or standing constraints: the worker already has them. For pause, give only the concrete
blocking reason or completed handoff. Both continue and pause are valid. No Markdown or extra
fields.
