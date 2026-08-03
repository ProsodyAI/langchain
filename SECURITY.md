# Security policy

## Reporting a vulnerability

Do not report security vulnerabilities in a public issue.

Email [security@prosodyai.app](mailto:security@prosodyai.app) with a description of the issue,
the affected version, reproduction steps, and the impact you observed. Do not include real
customer audio, API keys, or other sensitive data in the initial report.

The maintainers will acknowledge the report, validate it, and coordinate a fix and disclosure
timeline with the reporter.

## API keys and audio files

Keep `PROSODY_API_KEY` out of source control and logs. The LangChain tool requires an explicit
`allowed_audio_root` so an agent cannot request arbitrary files from the host filesystem. Treat
audio as sensitive application data and grant the tool access only to the directory it needs.
