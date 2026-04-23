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


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY is not set. Please add it in Secrets and restart.")
        st.stop()
    return genai.Client(api_key=api_key)


AUDIO_MIME_MAP = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aiff": "audio/aiff",
    # .mp4 from WhatsApp / voice recorders is audio-only — use audio/mp4
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


def guess_mime(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in AUDIO_MIME_MAP:
        return AUDIO_MIME_MAP[ext]
    if ext in VIDEO_MIME_MAP:
        return VIDEO_MIME_MAP[ext]
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def get_file_state(uploaded) -> str:
    """Return the file state as an uppercase string regardless of SDK version."""
    state = uploaded.state
    if state is None:
        return "ACTIVE"
    if isinstance(state, str):
        return state.upper()
    if hasattr(state, "name"):
        return state.name.upper()
    return str(state).upper()


def upload_to_gemini(client: genai.Client, file_bytes: bytes, filename: str):
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    mime = guess_mime(filename)
    uploaded = client.files.upload(
        file=tmp_path,
        config=types.UploadFileConfig(mime_type=mime, display_name=filename),
    )

    # Wait until the file is ACTIVE (poll up to 5 minutes)
    max_wait = 300
    waited = 0
    while get_file_state(uploaded) == "PROCESSING" and waited < max_wait:
        time.sleep(3)
        waited += 3
        uploaded = client.files.get(name=uploaded.name)

    if get_file_state(uploaded) == "FAILED":
        raise RuntimeError(
            f"Gemini could not process this file. "
            f"Make sure it is a valid audio or video recording."
        )

    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    return uploaded


def generate_with_retry(client: genai.Client, **kwargs) -> str:
    """Call generate_content with automatic retry on 429 rate-limit errors."""
    for attempt in range(5):
        try:
            response = client.models.generate_content(**kwargs)
            return response.text or ""
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = 60 * (attempt + 1)
                st.toast(f"Rate limit hit — waiting {wait}s before retry {attempt + 1}/5…")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini rate limit persists after 5 retries. Please wait a few minutes and try again.")


def transcribe(client: genai.Client, gemini_file) -> str:
    prompt = (
        "You are transcribing a meeting recording. Produce a clean, readable transcript "
        "of everything spoken. When possible, label distinct speakers as 'Speaker 1', "
        "'Speaker 2', etc., or use names if they introduce themselves. Include timestamps "
        "in [MM:SS] format at the start of each speaker turn. Do not summarize — provide "
        "the full transcript."
    )
    return generate_with_retry(
        client,
        model=MODEL_ID,
        contents=[gemini_file, prompt],
    )


def analyze_meeting(client: genai.Client, transcript: str) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
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
        "required": [
            "title",
            "summary",
            "participants",
            "key_decisions",
            "action_items",
            "next_steps",
        ],
    }

    prompt = (
        "Analyze the following meeting transcript and extract structured insights. "
        "Be specific and faithful to the transcript — do not invent details. If "
        "something is unknown, leave the field empty or omit the entry.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )

    raw = generate_with_retry(
        client,
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {
            "title": "Meeting",
            "summary": raw or "",
            "participants": [],
            "key_decisions": [],
            "action_items": [],
            "next_steps": [],
        }


def chat_about_meeting(client: genai.Client, transcript: str, history: list, question: str) -> str:
    system = (
        "You are an assistant that answers questions strictly about the meeting "
        "transcript provided below. Only use information from the transcript. If "
        "the answer is not in the transcript, say so plainly. Be concise and cite "
        "short quotes when helpful.\n\n"
        f"MEETING TRANSCRIPT:\n{transcript}"
    )

    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system),
    )
    return response.text or ""


# --- UI ---

st.set_page_config(page_title="Meeting Analyzer", page_icon="🎙️", layout="wide")

st.title("🎙️ Meeting Analyzer")
st.caption("Upload a meeting recording and let Gemini transcribe and summarize it for you.")

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "file_signature" not in st.session_state:
    st.session_state.file_signature = None

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader(
        "Audio or video file",
        type=sorted([e.lstrip(".") for e in SUPPORTED_EXTS]),
        accept_multiple_files=False,
    )
    analyze_btn = st.button("Analyze meeting", type="primary", use_container_width=True, disabled=uploaded is None)
    if st.button("Reset", use_container_width=True):
        for key in ["analysis", "transcript", "chat_history", "file_signature"]:
            st.session_state[key] = None if key != "chat_history" else []
        st.rerun()

    st.divider()
    st.markdown("**Tips**")
    st.markdown(
        "- Max upload size is 500 MB.\n"
        "- Longer recordings take longer to process.\n"
        "- The chat answers only from the meeting content."
    )

if analyze_btn and uploaded is not None:
    file_bytes = uploaded.getvalue()
    signature = (uploaded.name, len(file_bytes))
    if st.session_state.file_signature != signature:
        st.session_state.chat_history = []
    st.session_state.file_signature = signature

    client = get_client()
    try:
        with st.status("Processing your meeting...", expanded=True) as status:
            st.write("Uploading file to Gemini...")
            gemini_file = upload_to_gemini(client, file_bytes, uploaded.name)

            st.write("Transcribing audio...")
            transcript = transcribe(client, gemini_file)
            st.session_state.transcript = transcript

            st.write("Extracting summary, action items, decisions, and participants...")
            analysis = analyze_meeting(client, transcript)
            st.session_state.analysis = analysis

            status.update(label="Done!", state="complete")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

if st.session_state.analysis:
    a = st.session_state.analysis

    st.subheader(a.get("title") or "Meeting")

    tabs = st.tabs(["Summary", "Action Items", "Decisions", "Participants", "Next Steps", "Transcript", "Chat"])

    with tabs[0]:
        st.markdown(a.get("summary") or "_No summary available._")

    with tabs[1]:
        items = a.get("action_items") or []
        if not items:
            st.info("No action items identified.")
        else:
            for item in items:
                task = item.get("task", "")
                owner = item.get("owner") or "Unassigned"
                due = item.get("due") or ""
                meta = f"**{owner}**" + (f" • _{due}_" if due else "")
                st.markdown(f"- {task}  \n  {meta}")

    with tabs[2]:
        decisions = a.get("key_decisions") or []
        if not decisions:
            st.info("No key decisions identified.")
        else:
            for d in decisions:
                st.markdown(f"- {d}")

    with tabs[3]:
        people = a.get("participants") or []
        if not people:
            st.info("No participants identified.")
        else:
            for p in people:
                name = p.get("name", "")
                role = p.get("role") or ""
                st.markdown(f"- **{name}**" + (f" — {role}" if role else ""))

    with tabs[4]:
        steps = a.get("next_steps") or []
        if not steps:
            st.info("No next steps identified.")
        else:
            for s in steps:
                st.markdown(f"- {s}")

    with tabs[5]:
        st.text_area("Full transcript", st.session_state.transcript or "", height=500)
        st.download_button(
            "Download transcript",
            data=(st.session_state.transcript or "").encode("utf-8"),
            file_name="transcript.txt",
            mime="text/plain",
        )

    with tabs[6]:
        st.markdown("Ask anything about this meeting. Answers come **only** from the transcript.")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask a question about the meeting...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    client = get_client()
                    try:
                        answer = chat_about_meeting(
                            client,
                            st.session_state.transcript or "",
                            st.session_state.chat_history[:-1],
                            question,
                        )
                    except Exception as e:
                        answer = f"Sorry, I couldn't answer that: {e}"
                    st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a meeting recording in the sidebar and click **Analyze meeting** to get started.")
