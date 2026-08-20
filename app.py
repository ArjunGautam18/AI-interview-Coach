"""
AI Interview Coach
-------------------
A 3-stage AI pipeline wrapped in a Streamlit app:

	Audio answer --(Speech-to-Text)--> Transcript
				 --(NLP evaluation)--> Score + Feedback
				 --(Text Generation)--> Improved answer

Stage 1 (Speech-to-Text):  openai/whisper-base  (via HuggingFace transformers)
Stage 2 (Evaluation):      google/flan-t5-base  (scores + feedback)
Stage 3 (Generation):      google/flan-t5-base  (suggests a better answer)
"""

from io import BytesIO
import re
from datetime import datetime

import librosa
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


DEMO_QUESTION = "Tell me about yourself."
DEMO_ANSWER = (
	"Um so yeah, my name is uh, well, I studied computer science in college and then "
	"I worked at a couple of companies doing like coding stuff and testing stuff. "
	"I think I'm pretty good at solving problems and I work well with teams I guess. "
	"I don't know, I just really want a job where I can learn more things and grow "
	"my career I think."
)

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="AI Interview Coach", page_icon="🎤", layout="centered")
st.title("🎤 AI Interview Coach")
st.caption("Answer a question out loud → get transcribed, scored, and improved by AI")


# ----------------------------------------------------------------------
# Cache model loading so it only happens once per session
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_whisper():
	return pipeline("automatic-speech-recognition", model="openai/whisper-base")


@st.cache_resource(show_spinner=False)
def load_evaluator():
	tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
	model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
	return tokenizer, model


def generate_text(prompt: str, **generation_options) -> str:
	tokenizer, model = load_evaluator()
	inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
	outputs = model.generate(**inputs, **generation_options)
	return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ----------------------------------------------------------------------
# STAGE 1: Speech-to-Text
# ----------------------------------------------------------------------
def transcribe_audio(audio_file) -> str:
	audio_file.seek(0)
	audio_bytes = audio_file.read()
	audio_waveform, _ = librosa.load(BytesIO(audio_bytes), sr=16000, mono=True)
	asr = load_whisper()
	result = asr({"raw": audio_waveform, "sampling_rate": 16000})
	return result["text"].strip()


# ----------------------------------------------------------------------
# STAGE 2: NLP Evaluation — score the answer
# ----------------------------------------------------------------------
def evaluate_answer(question: str, answer: str) -> str:
	prompt = (
		f"You are an interview coach. Question: {question}\n"
		f"Candidate's answer: {answer}\n"
		f"Rate the answer on Relevance (1-10), Clarity (1-10), and Grammar (1-10). "
		f"Then give one short sentence of feedback. "
		f"Format exactly like: Relevance: X, Clarity: X, Grammar: X. Feedback: ..."
	)
	return generate_text(prompt, max_new_tokens=120, do_sample=False)


def fallback_scores(question: str, answer: str) -> dict:
	"""Provide usable estimates when the text model does not return numbers."""
	question_words = set(re.findall(r"[a-z]+", question.lower()))
	answer_words = set(re.findall(r"[a-z]+", answer.lower()))
	overlap = len(question_words & answer_words) / max(len(question_words), 1)
	filler_count = len(re.findall(r"\b(um|uh|like|yeah|i guess|i don't know)\b", answer.lower()))
	sentence_count = max(len(re.findall(r"[.!?]", answer)), 1)
	average_sentence_length = len(answer.split()) / sentence_count

	return {
		"Relevance": max(1, min(10, round(5 + overlap * 5))),
		"Clarity": max(1, min(10, round(8 - filler_count * 0.7 - max(average_sentence_length - 25, 0) * 0.1))),
		"Grammar": max(1, min(10, round(8 - filler_count * 0.5 - (1 if answer and answer[-1] not in ".!?" else 0)))),
	}


def parse_scores(eval_text: str, question: str, answer: str) -> dict:
	"""Parse model scores and fill missing values with local estimates."""
	scores = {}
	for label in ["Relevance", "Clarity", "Grammar"]:
		match = re.search(rf"{label}\s*[:=-]\s*(?:score\s*)?(\d{{1,2}})", eval_text, re.IGNORECASE)
		scores[label] = int(match.group(1)) if match else None

	fallback = fallback_scores(question, answer)
	return {
		label: max(1, min(10, value if value is not None else fallback[label]))
		for label, value in scores.items()
	}


# ----------------------------------------------------------------------
# STAGE 3: Text Generation — suggest an improved answer
# ----------------------------------------------------------------------
def generate_improved_answer(question: str, answer: str) -> str:
	prompt = (
		f"Interview question: {question}\n"
		f"Candidate's original answer: {answer}\n"
		f"Rewrite this as a stronger, more confident, well-structured interview answer "
		f"in 3-4 sentences."
	)
	return generate_text(prompt, max_new_tokens=180, do_sample=True, temperature=0.8, top_p=0.9)


# ----------------------------------------------------------------------
# UI FLOW
# ----------------------------------------------------------------------
st.subheader("1. Interview question")
default_questions = [
	"Tell me about yourself.",
	"What is your biggest weakness?",
	"Why should we hire you?",
	"Describe a challenge you overcame at work.",
	"Write your own question",
]
choice = st.selectbox("Pick a question or write your own:", default_questions)
if choice == "Write your own question":
	question = st.text_input("Type your interview question:")
else:
	question = choice

st.subheader("2. Your answer")
answer_mode = st.radio("Choose how to provide your answer:", ["Audio upload", "Text answer", "Built-in demo"], horizontal=True)
use_demo = answer_mode == "Built-in demo"
text_answer = ""
if answer_mode == "Text answer":
	text_answer = st.text_area(
		"Type your interview answer:",
		height=160,
		placeholder="Write the answer you would give in an interview...",
	)
audio_file = None
if answer_mode == "Audio upload":
	audio_file = st.file_uploader(
		"Upload an audio recording of your answer (wav/mp3/m4a)",
		type=["wav", "mp3", "m4a"],
	)
if answer_mode == "Audio upload":
	st.caption("Tip: record yourself answering the question on your phone, then upload the file here.")

if use_demo and question:
	st.info("Demo answer loaded. Click Analyze My Answer to see the full coaching pipeline.")

answer_ready = audio_file is not None if answer_mode == "Audio upload" else bool(text_answer.strip()) or use_demo
if answer_ready and question:
	if audio_file is not None:
		st.audio(audio_file)

	if st.button("🎯 Analyze My Answer"):
		# Stage 1
		if use_demo:
			transcript = DEMO_ANSWER
		elif answer_mode == "Text answer":
			transcript = text_answer.strip()
		else:
			with st.spinner("Transcribing your answer..."):
				transcript = transcribe_audio(audio_file)
		st.subheader("📝 Transcript")
		st.write(transcript)

		if not transcript:
			st.warning("Couldn't detect any speech in the audio. Try a clearer recording.")
		else:
			# Stage 2
			with st.spinner("Evaluating your answer..."):
				eval_text = evaluate_answer(question, transcript)
				scores = parse_scores(eval_text, question, transcript)

			st.subheader("📊 Scores & Feedback")
			overall_score = round(sum(scores.values()) / len(scores), 1)
			st.metric("Overall score", f"{overall_score}/10")
			st.progress(overall_score / 10)
			cols = st.columns(3)
			for col, (label, value) in zip(cols, scores.items()):
				col.metric(label, f"{value}/10" if value else "N/A")
			st.write(eval_text)

			# Stage 3
			with st.spinner("Drafting a stronger answer..."):
				improved = generate_improved_answer(question, transcript)
			st.subheader("✨ Suggested Improved Answer")
			st.write(improved)

			report = (
				"AI Interview Coach Report\n"
				f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
				f"Question\n{question}\n\n"
				f"Transcript / Answer\n{transcript}\n\n"
				f"Scores\n"
				f"Relevance: {scores['Relevance']}/10\n"
				f"Clarity: {scores['Clarity']}/10\n"
				f"Grammar: {scores['Grammar']}/10\n"
				f"Overall: {overall_score}/10\n\n"
				f"Feedback\n{eval_text}\n\n"
				f"Suggested Improved Answer\n{improved}\n"
			)
			st.download_button(
				"⬇️ Download coaching report",
				data=report,
				file_name="ai_interview_coach_report.txt",
				mime="text/plain",
			)
else:
	if answer_mode == "Audio upload":
		st.info("Select a question and upload your audio answer to get started.")
	elif answer_mode == "Text answer":
		st.info("Select a question and type your answer to get started.")