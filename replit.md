# Workspace

## Overview

Streamlit Meeting Analyzer that uses Google Gemini to transcribe uploaded meeting
audio/video files and extract a summary, action items, key decisions, participants,
and next steps. Includes a chat that answers questions strictly from the meeting
transcript.

## Stack

- **Language**: Python 3.11
- **UI**: Streamlit
- **AI**: Google Gemini (`google-genai` SDK, model `gemini-2.0-flash`)

## Files

- `app.py` — Streamlit application (upload, transcribe, analyze, chat)
- `.streamlit/config.toml` — Streamlit server config (port 5000, max upload 500 MB)

## Secrets

- `GEMINI_API_KEY` — required

## Run

- Workflow: `Streamlit App` runs `streamlit run app.py --server.port 5000`
