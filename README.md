# Meeting Analyzer

A Streamlit app that uses Google Gemini to transcribe meeting audio/video and extract:

- Summary
- Action items (with owners and due dates)
- Key decisions
- Participants
- Next steps

It also includes a chat that answers questions strictly from the meeting transcript.

## Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
streamlit run app.py
```

Get a free Gemini API key at https://aistudio.google.com/apikey.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and sign in.
3. Click **New app**, select your repo, branch `main`, and main file `app.py`.
4. Under **Advanced settings → Secrets**, add:

   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```

5. Click **Deploy**.
