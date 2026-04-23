import os
import json
import time
import tempfile
import mimetypes
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types


MODEL_ID = "gemini-2.0-flash-lite"

SUPPORTED_AUDIO = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".aiff"}
SUPPORTED_VIDEO = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg", ".3gp"}
SUPPORTED_EXTS = SUPPORTED_AUDIO | SUPPORTED_VIDEO

AUDIO_MIME_MAP = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aiff": "audio/aiff",
    ".mp4": "audio/mp4",
}

VIDEO_MIME_MAP = {
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".3gp": "video/3gpp",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "transcript": {"type": "string"},
        "summary": {"type": "string"},
        "participants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "key_decisions": {"type": "array", "items": {"type": "string"}},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "owner": {"type": "string"},
                    "due": {"type": "string"},
                },
                "required": ["task"],
            },
        },
        "next_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "transcript", "summary", "participants",
                 "key_decisions", "action_items", "next_steps"],
}

PROMPT = (
    "Listen to this meeting recording carefully and return a JSON object with these fields:\n"
    "- title: a short descriptive title for the meeting\n"
    "- transcript: full verbatim transcript with speaker labels (Speaker 1, Speaker 2, etc.) "
    "and [MM:SS] timestamps at each turn\n"
    "- summary: concise summary of the meeting in 3-5 sentences\n"
    "- participants: list of speakers with name and role if mentioned\n"
    "- key_decisions: list of decisions made\n"
    "- action_items: list of tasks with owner and due date if mentioned\n"
    "- next_steps: what happens after this meeting\n"
    "Be faithful to what was said. Do not invent details."
)


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY is not set. Please add it in Secrets and restart.")
        st.stop()
    return genai.Client(api_key=api_key)


def guess_mime(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in AUDIO_MIME_MAP:
        return AUDIO_MIME_MAP[ext]
    if ext in VIDEO_MIME_MAP:
        return VIDEO_MIME_MAP[ext]
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def get_file_state(f) -> str:
    state = f.state
    if state is None:
        return "ACTIVE"
    if isinstance(state, str):
        return state.upper()
    if hasattr(state, "name"):
        return state.name.upper()
    return str(state).upper()


def upload_file(client: genai.Client, file_bytes: bytes, filename: str):
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    uploaded = client.files.upload(
        file=tmp_path,
        config=types.UploadFileConfig(mime_type=guess_mime(filename), display_name=filename),
    )

    max_wait, waited = 300, 0
    while get_file_state(uploaded) == "PROCESSING" and waited < max_wait:
        time.sleep(3)
        waited += 3
        uploaded = client.files.get(name=uploaded.name)

    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    if get_file_state(uploaded) == "FAILED":
        raise RuntimeError("Gemini could not process this file. Make sure it is a valid audio or video recording.")

    return uploaded


def analyze_in_one_call(client: genai.Client, gemini_file) -> dict:
    """Single API call: transcribe + analyze together. Saves tokens and time."""
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[gemini_file, PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SCHEMA,
                ),
            )
            raw = response.text or "{}"
            return json.loads(raw)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = 65
                st.toast(f"Rate limit — waiting {wait}s then retrying ({attempt + 1}/4)…")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Rate limit persists. Please wait a few minutes and try again.")


def chat_about_meeting(client: genai.Client, transcript: str, history: list, question: str) -> str:
    system = (
        "You answer questions strictly based on the meeting transcript below. "
        "Only use what is in the transcript. Be concise.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system),
            )
            return response.text or ""
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(65)
            else:
                raise
    return "Rate limit hit. Please wait a moment and try again."


# --- UI ---

st.set_page_config(page_title="Meeting Analyzer", page_icon="🎙️", layout="wide")
st.title("🎙️ Meeting Analyzer")
st.caption("Upload a meeting recording — Gemini transcribes and analyzes it in one step.")

for key, default in [("result", None), ("chat_history", []), ("file_sig", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader(
        "Audio or video file",
        type=sorted([e.lstrip(".") for e in SUPPORTED_EXTS]),
        accept_multiple_files=False,
    )
    analyze_btn = st.button("Analyze meeting", type="primary",
                            use_container_width=True, disabled=uploaded is None)
    if st.button("Reset", use_container_width=True):
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.file_sig = None
        st.rerun()
    st.divider()
    st.markdown("**Tips**\n- Max 500 MB\n- Longer files take more time\n- Chat answers only from the meeting")

if analyze_btn and uploaded is not None:
    sig = (uploaded.name, uploaded.size)
    if st.session_state.file_sig != sig:
        st.session_state.chat_history = []
    st.session_state.file_sig = sig

    client = get_client()
    try:
        with st.status("Analyzing your meeting…", expanded=True) as status:
            st.write("Uploading file…")
            gemini_file = upload_file(client, uploaded.getvalue(), uploaded.name)
            st.write("Transcribing and analyzing (one step)…")
            result = analyze_in_one_call(client, gemini_file)
            st.session_state.result = result
            status.update(label="Done!", state="complete")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

r = st.session_state.result
if r:
    st.subheader(r.get("title") or "Meeting")
    tabs = st.tabs(["Summary", "Action Items", "Decisions", "Participants", "Next Steps", "Transcript", "Chat"])

    with tabs[0]:
        st.markdown(r.get("summary") or "_No summary._")

    with tabs[1]:
        items = r.get("action_items") or []
        if not items:
            st.info("No action items identified.")
        else:
            for item in items:
                owner = item.get("owner") or "Unassigned"
                due = item.get("due") or ""
                st.markdown(f"- {item.get('task', '')}  \n  **{owner}**" + (f" • _{due}_" if due else ""))

    with tabs[2]:
        decisions = r.get("key_decisions") or []
        if not decisions:
            st.info("No key decisions identified.")
        else:
            for d in decisions:
                st.markdown(f"- {d}")

    with tabs[3]:
        people = r.get("participants") or []
        if not people:
            st.info("No participants identified.")
        else:
            for p in people:
                role = p.get("role") or ""
                st.markdown(f"- **{p.get('name', '')}**" + (f" — {role}" if role else ""))

    with tabs[4]:
        steps = r.get("next_steps") or []
        if not steps:
            st.info("No next steps identified.")
        else:
            for s in steps:
                st.markdown(f"- {s}")

    with tabs[5]:
        transcript = r.get("transcript") or ""
        st.text_area("Full transcript", transcript, height=500)
        st.download_button("Download transcript", data=transcript.encode("utf-8"),
                           file_name="transcript.txt", mime="text/plain")

    with tabs[6]:
        st.markdown("Ask anything — answers come **only** from this meeting.")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        question = st.chat_input("Ask a question about the meeting…")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    client = get_client()
                    try:
                        answer = chat_about_meeting(
                            client, r.get("transcript") or "",
                            st.session_state.chat_history[:-1], question,
                        )
                    except Exception as e:
                        answer = f"Error: {e}"
                    st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a recording in the sidebar and click **Analyze meeting** to get started.")
