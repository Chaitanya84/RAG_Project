import os
import sys
import random

# Portable path: resolve relative to this file's location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from AnswerAll import load_rag_resources, generate_quiz
from config import CHUNKS_CSV, EMBEDDINGS_PATH, logger

# NOTE: st.set_page_config() is only called in the main app.py entry point.

# ---------------------------------------------------------------------------
# Require login
# ---------------------------------------------------------------------------
if "user_logged_in" not in st.session_state or not st.session_state.user_logged_in:
    st.error("You are not logged in. Please go to the main page and log in first.")
    if st.button("Go to Login Page"):
        st.switch_page("app.py")
    st.stop()

# ---------------------------------------------------------------------------
# Load knowledge base
# ---------------------------------------------------------------------------
chunks_csv = st.session_state.get("active_chunks_csv") or CHUNKS_CSV
embeddings_path = st.session_state.get("active_embeddings_path") or EMBEDDINGS_PATH

if not os.path.exists(chunks_csv) or not os.path.exists(embeddings_path):
    st.warning(
        "No processed PDF data found. Please process a PDF directory on the **MainPage** first."
    )
    st.stop()


@st.cache_resource
def get_rag_resources(_chunks_csv: str, _embeddings_path: str):
    """Load RAG resources; cached so they are only read once per unique path pair."""
    return load_rag_resources(chunks_csv=_chunks_csv, embeddings_path=_embeddings_path)


pages_and_chunks, embeddings = get_rag_resources(chunks_csv, embeddings_path)

# ---------------------------------------------------------------------------
# Sidebar — active knowledge base info
# ---------------------------------------------------------------------------
processed_dir = st.session_state.get("last_processed_dir")
processed_files = st.session_state.get("processed_files", [])
if processed_dir:
    with st.sidebar:
        st.markdown("### 📂 Active Knowledge Base")
        st.markdown(f"**Directory:** `{processed_dir}`")
        if processed_files:
            st.markdown(f"**PDFs loaded:** {len(processed_files)}")
            with st.expander("Show files"):
                for pf in processed_files:
                    st.markdown(f"- `{pf}`")

# ---------------------------------------------------------------------------
# Session state for quiz
# ---------------------------------------------------------------------------
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🧠 Knowledge Quiz")
st.write(f"Logged in as: {st.session_state.user_email}")
st.markdown(
    "Test your understanding of the knowledge base! "
    "Pick a source document or use random passages, choose difficulty and number of questions, then take the quiz."
)

# ---------------------------------------------------------------------------
# Build topic options from the knowledge base
# ---------------------------------------------------------------------------
# Extract unique source files from the loaded chunks
source_files = sorted(set(
    chunk.get("source_file", "unknown")
    for chunk in pages_and_chunks
    if chunk.get("source_file")
))

# Build topic choices: "All documents", each source file, and "Random passages"
topic_choices = ["📚 All documents — random sample"] + [
    f"📄 {sf}" for sf in source_files
] + ["🎲 Random passages"]

# ---------------------------------------------------------------------------
# Quiz configuration
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    topic_selection = st.selectbox(
        "📌 Quiz Source",
        options=topic_choices,
        index=0,
        help="Choose which document(s) to generate quiz questions from.",
    )
with col2:
    difficulty = st.selectbox(
        "⚙️ Difficulty",
        options=["Easy", "Medium", "Hard"],
        index=1,
    )
with col3:
    num_questions = st.selectbox(
        "🔢 Questions",
        options=[3, 5, 7, 10],
        index=1,
    )

# ---------------------------------------------------------------------------
# Generate quiz
# ---------------------------------------------------------------------------
if st.button("🚀 Generate Quiz", type="primary", use_container_width=True):
    # Reset previous quiz state
    st.session_state.quiz_questions = []
    st.session_state.quiz_submitted = False
    st.session_state.quiz_answers = {}
    st.session_state.quiz_score = 0

    # Determine which chunks to use based on the selection
    if topic_selection == "🎲 Random passages":
        # Pick random chunks directly from the knowledge base
        sample_size = min(max(num_questions * 2, 5), 15, len(pages_and_chunks))
        selected_chunks = random.sample(pages_and_chunks, sample_size)
        topic_hint = "general content from the knowledge base"
    elif topic_selection == "📚 All documents — random sample":
        sample_size = min(max(num_questions * 2, 5), 15, len(pages_and_chunks))
        selected_chunks = random.sample(pages_and_chunks, sample_size)
        topic_hint = "broad coverage across all documents in the knowledge base"
    else:
        # Specific source file selected — filter chunks from that file
        selected_file = topic_selection.replace("📄 ", "")
        file_chunks = [
            c for c in pages_and_chunks
            if c.get("source_file") == selected_file
        ]
        if not file_chunks:
            st.error(f"No chunks found for `{selected_file}`.")
            st.stop()
        sample_size = min(max(num_questions * 2, 5), 15, len(file_chunks))
        selected_chunks = random.sample(file_chunks, sample_size)
        topic_hint = f"content from {selected_file}"

    with st.spinner(f"Generating {num_questions} {difficulty.lower()} questions…"):
        try:
            questions = generate_quiz(
                topic=topic_hint,
                pages_and_chunks=pages_and_chunks,
                embeddings=embeddings,
                num_questions=num_questions,
                difficulty=difficulty,
                context_override=selected_chunks,
            )
            st.session_state.quiz_questions = questions
        except Exception as e:
            logger.error("Quiz generation failed: %s", e)
            st.error(f"Failed to generate quiz: {e}")
            st.stop()

    st.rerun()

# ---------------------------------------------------------------------------
# Render quiz
# ---------------------------------------------------------------------------
questions = st.session_state.quiz_questions

if questions and not st.session_state.quiz_submitted:
    st.divider()
    st.subheader(f"📝 Quiz — {len(questions)} Questions")

    with st.form("quiz_form"):
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i + 1}.** {q['question']}")
            options = q["options"]
            choices = [f"{key}: {val}" for key, val in options.items()]
            selected = st.radio(
                f"Select your answer for Q{i + 1}:",
                options=choices,
                index=None,
                key=f"q_{i}",
                label_visibility="collapsed",
            )
            # Store just the letter (A/B/C/D)
            if selected:
                st.session_state.quiz_answers[i] = selected.split(":")[0].strip()
            st.markdown("---")

        submitted = st.form_submit_button(
            "✅ Submit Answers", type="primary", use_container_width=True
        )

        if submitted:
            # Check if all questions are answered
            unanswered = [
                i + 1 for i in range(len(questions))
                if i not in st.session_state.quiz_answers
            ]
            if unanswered:
                st.warning(
                    f"Please answer all questions. Missing: {', '.join(f'Q{n}' for n in unanswered)}"
                )
            else:
                # Calculate score
                score = 0
                for i, q in enumerate(questions):
                    if st.session_state.quiz_answers.get(i) == q["correct"]:
                        score += 1
                st.session_state.quiz_score = score
                st.session_state.quiz_submitted = True
                st.rerun()

# ---------------------------------------------------------------------------
# Show results
# ---------------------------------------------------------------------------
if questions and st.session_state.quiz_submitted:
    st.divider()
    score = st.session_state.quiz_score
    total = len(questions)
    pct = (score / total) * 100

    # Score banner
    if pct >= 80:
        st.success(f"🎉 Excellent! You scored **{score}/{total}** ({pct:.0f}%)")
    elif pct >= 50:
        st.warning(f"👍 Good effort! You scored **{score}/{total}** ({pct:.0f}%)")
    else:
        st.error(f"📖 Keep learning! You scored **{score}/{total}** ({pct:.0f}%)")

    # Progress bar
    st.progress(pct / 100)

    # Detailed review
    st.subheader("📋 Review")
    for i, q in enumerate(questions):
        user_ans = st.session_state.quiz_answers.get(i, "—")
        correct_ans = q["correct"]
        is_correct = user_ans == correct_ans

        icon = "✅" if is_correct else "❌"
        st.markdown(f"**{icon} Q{i + 1}.** {q['question']}")

        # Show all options, highlight correct and user's choice
        for key, val in q["options"].items():
            if key == correct_ans and key == user_ans:
                st.markdown(f"&emsp;🟢 **{key}: {val}** ← Your answer (Correct!)")
            elif key == correct_ans:
                st.markdown(f"&emsp;🟢 **{key}: {val}** ← Correct answer")
            elif key == user_ans:
                st.markdown(f"&emsp;🔴 ~~{key}: {val}~~ ← Your answer")
            else:
                st.markdown(f"&emsp;⚪ {key}: {val}")

        with st.expander("💡 Explanation"):
            st.info(q["explanation"])

        st.markdown("---")

    # Action buttons
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Take Another Quiz", use_container_width=True):
            st.session_state.quiz_questions = []
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = {}
            st.session_state.quiz_score = 0
            st.rerun()
    with col_b:
        if st.button("🗑️ Clear & Go to Chat", use_container_width=True):
            st.session_state.quiz_questions = []
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = {}
            st.session_state.quiz_score = 0
            st.switch_page("pages/chatPage.py")

# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------
with st.sidebar:
    if st.button("Logout"):
        for key in [
            "user_logged_in", "user_email", "quiz_questions",
            "quiz_submitted", "quiz_answers", "quiz_score",
        ]:
            st.session_state.pop(key, None)
        st.switch_page("app.py")
