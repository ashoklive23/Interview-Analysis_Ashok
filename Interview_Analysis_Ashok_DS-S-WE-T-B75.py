import streamlit as st
import tempfile
import os
import requests
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from pydub import AudioSegment
from docx import Document
import whisper
from streamlit_lottie import st_lottie

nltk.download('punkt', quiet=True)

def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_interview = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json")
lottie_bot = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_m9cnrcf3.json")

vader = SentimentIntensityAnalyzer()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Fastest model
@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

def transcribe_audio(uploaded_audio):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    audio = AudioSegment.from_file(uploaded_audio)
    audio.export(temp.name, format="wav")
    model = load_whisper()
    result = model.transcribe(temp.name)
    os.remove(temp.name)
    return result["text"]

def read_docx(uploaded_file):
    doc = Document(uploaded_file)
    return '\n'.join([p.text for p in doc.paragraphs])

def keyword_score(text, expected_keywords):
    found = [kw.lower() for kw in expected_keywords if kw.lower() in text.lower()]
    return len(found), found

st.set_page_config(page_title="MentorFlow | AI Interview Analyzer", layout="wide")
st.markdown("""
<style>
    .badge {display:inline-block;border-radius:2em;padding:0.35em 1.5em;font-weight:bold;
            color:#fff;margin:0.2em;font-size:1.15em;}
    .excellent {background:linear-gradient(90deg,#2196f3,#00c853);}
    .good {background:linear-gradient(90deg,#43a047,#fbc02d);}
    .average {background:linear-gradient(90deg,#fbc02d,#e53935);}
    .poor {background:linear-gradient(90deg,#ef5350,#455a64);}
    section.css-1wbqy5l {max-width:1100px !important;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    if lottie_interview is not None:
        st_lottie(lottie_interview, speed=1, loop=True, quality="medium", height=100)
    else:
        st.warning("Opening animation unavailable.")
    st.markdown('<div style="font-size:1.9em;font-weight:bold;color:#2e5cfc;">MentorFlow</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.05em; color:#444;">AI Interview Analyzer</div>', unsafe_allow_html=True)
    st.markdown("---")
    input_type = st.radio("Choose Input Type", [
        "Audio (.mp3/.wav/.m4a)",
        "Text (.txt)",
        "Word (.docx)",
        "Paste Text",
        "AI Meeting BOT"
    ])
    text_input = ""
    uploaded_file = None
    analyze_btn = None
    # AI Meeting Bot
    if input_type == "AI Meeting BOT":
        st.markdown("### 🤖 AI Meeting Bot Join")
        meeting_platform = st.selectbox(
            "Meeting Platform", ["Microsoft Teams", "Google Meet", "Zoom", "Skype", "Webex"]
        )
        meeting_id = st.text_input("Enter Meeting Link or Meeting ID")
        if lottie_bot is not None:
            st_lottie(lottie_bot, speed=1, loop=True, height=90, quality="medium", key="meetbot")
        else:
            st.warning("Bot animation unavailable.")
        bot_btn = st.button(f"Join & Analyze {meeting_platform} Meeting", key="joinbot")
        if bot_btn:
            if meeting_id.strip():
                st.success(f"MentorFlow Bot is joining {meeting_platform} meeting: {meeting_id}")
                st.info("Demo: In production, transcript/audio would be analyzed live!")
            else:
                st.error("Please enter a valid meeting link/ID.")
    else:
        industry = st.selectbox(
            "Interview Domain",
            ['IT', 'Supply Chain', 'Testing', 'BPO', 'Manufacturing', 'Banking', 'Healthcare', 'Retail', 'Marketing', 'Education', 'Other']
        )
        subdomain_options = {
            "IT": ["Python", "SQL", "Java", "Data Science", "Cloud", "Cybersecurity", "DevOps","Frontend","Backend"],
            "Supply Chain": ["Procurement", "Sourcing", "Logistics", "Warehousing", "Inventory", "Planning"],
            "Testing": ["Manual", "Automation", "Performance", "Regression", "Security"],
            "BPO": ["Customer Support", "Technical Support", "Finance & Accounting", "Data Entry"],
            "Manufacturing": ["Production", "Quality", "Maintenance", "Safety", "Process Engineering"],
            "Banking": ["Retail Banking", "Risk", "Treasury", "Corporate", "Credit"],
            "Healthcare": ["Clinical", "Pharma", "MedTech", "Nursing", "Administration"],
            "Retail": ["Store Ops", "Merchandising", "Supply", "Buying", "E-commerce"],
            "Marketing": ["Digital", "Brand", "Content", "Market Research", "PR"],
            "Education": ["Teaching", "Counseling", "Administration", "EdTech"],
            "Other": ["Other"]
        }
        subdomain = st.selectbox(
            "Subdomain/Role",
            subdomain_options.get(industry, ["General"])
        )
        exp_level = st.radio("Candidate Type", ["Fresher", "Experienced"], index=0)
        round_type = st.selectbox("Round Type", ['Interview', 'Group Discussion', 'Mock'])

        if input_type == 'Paste Text':
            text_input = st.text_area("Paste transcript here:", height=160)
        else:
            upload_types = {
                'Audio (.mp3/.wav/.m4a)': ['mp3', 'wav', 'm4a'],
                'Text (.txt)': ['txt'],
                'Word (.docx)': ['docx']
            }
            uploaded_file = st.file_uploader(
                "Upload File",
                type=upload_types[input_type] if input_type in upload_types else None
            )
        analyze_btn = st.button("🎙️ Analyze Now", use_container_width=True)

st.title("MentorFlow Interview Analytics")
st.write("AI-powered evaluation & feedback for interviews and group discussions.")

if analyze_btn:
    transcript = ""
    # AUDIO: Check for short file before transcribing!
    if input_type == 'Audio (.mp3/.wav/.m4a)' and uploaded_file:
        # Speed: Limit to 30 seconds!
        audio = AudioSegment.from_file(uploaded_file)
        duration_sec = len(audio) / 1000
        if duration_sec > 30:
            st.error("Audio > 30s. Please upload a shorter audio file for fastest results.")
            st.stop()
        st.info(":blue[Note: Only audio <=30s allowed for instant result! Whisper Tiny engine used.]")
        transcript = transcribe_audio(uploaded_file)
        st.success("Audio successfully transcribed!")
    elif input_type == 'Text (.txt)' and uploaded_file:
        transcript = uploaded_file.read().decode('utf-8', errors='ignore')
    elif input_type == 'Word (.docx)' and uploaded_file:
        transcript = read_docx(uploaded_file)
    elif input_type == 'Paste Text' and text_input.strip():
        transcript = text_input.strip()
    else:
        st.error("Please upload or paste transcript.")
        st.stop()
    if not transcript or len(transcript.strip()) < 10:
        st.warning("Insufficient data for analysis.")
        st.stop()

    domain_keywords = {
        "Python": ["function", "class", "list comprehension", "lambda", "pandas", "inheritance", "decorator"],
        "SQL": ["join", "index", "group by", "subquery", "transaction", "primary key", "foreign key"],
        "Procurement": ["supplier", "rfq", "negotiation", "contract", "purchase order", "bid", "sourcing"],
        "Manual": ["test case", "defect", "bug", "test plan", "execution", "report", "step"],
        "Customer Support": ["ticket", "sla", "escalation", "customer", "resolution", "support"],
        "Production": ["line", "downtime", "output", "maintenance", "efficiency"],
        "Retail": ["inventory", "stock", "sku", "replenishment", "planogram"],
        "Digital": ["seo", "adwords", "analytics", "content", "social media", "campaign"]
    }
    relevant_keywords = domain_keywords.get(subdomain, domain_keywords.get(industry, []))
    lines = transcript.strip().split('\n')
    speaker_data = {}
    for line in lines:
        if ':' in line:
            speaker, content = line.split(':', 1)
            speaker = speaker.strip()
            content = content.strip()
            speaker_data.setdefault(speaker, []).append(content)
        elif line.strip():
            speaker_data.setdefault("Unknown", []).append(line.strip())

    speaker_scores, analysis_cards, knowledge_scores = [], [], []
    for speaker, sentences in speaker_data.items():
        full_text = " ".join(sentences)
        if len(full_text) < 10:
            continue
        vader_scores = vader.polarity_scores(full_text)
        tb_sentiment = TextBlob(full_text).sentiment
        confidence = max(0, min(1, vader_scores['pos'] - vader_scores['neg'] + (1-tb_sentiment.subjectivity)))
        empathy_phrases = ['I understand', 'I appreciate', 'thank you', 'good question', 'that makes sense', 'let me help']
        empathy_score = sum(phrase in full_text.lower() for phrase in empathy_phrases)
        tokens = nltk.word_tokenize(full_text)
        filler_words = ['um', 'uh', 'like', 'you know']
        num_fillers = sum(tokens.count(w) for w in filler_words)
        avg_sent_len = len(tokens) // max(1, len(nltk.sent_tokenize(full_text)))
        tone = 'Positive' if vader_scores['compound'] > 0.2 else 'Negative' if vader_scores['compound'] < -0.2 else 'Neutral'
        summary = summarizer(full_text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
        keywords = [w for w in set(tokens) if w.isalpha() and len(w) > 5]
        score = max(0, min(10, (
            (vader_scores['compound']+1)*3 +
            3*confidence +
            2*min(empathy_score,1) -
            2*min(num_fillers,2)
        )))
        num_kw, found_kw = keyword_score(full_text, relevant_keywords) if relevant_keywords else (0,[])
        max_possible = len(relevant_keywords) if relevant_keywords else 1
        know_score = (num_kw / max_possible) * 10 if max_possible else 6
        knowledge_scores.append(know_score)
        analysis_cards.append({
            "speaker": speaker, "tone": tone, "score": score,
            "confidence": confidence, "empathy": empathy_score,
            "clarity": f"Filler words: {num_fillers}, Avg. sent len: {avg_sent_len}",
            "summary": summary, "keywords": ', '.join(keywords[:10]),
            "knowledge_score": know_score, "knowledge_found": found_kw
        })
        speaker_scores.append(score)

    if speaker_scores:
        avg_score = sum(speaker_scores)/len(speaker_scores)
        avg_know = sum(knowledge_scores)/len(knowledge_scores) if knowledge_scores else 6
        if avg_know < 5:
            perf_badge = '<span class="badge poor">❌ Low Knowledge - Not Recommended</span>'
            decision = "Do NOT Select"
            decision_msg = "❌ AI recommends NOT selecting this candidate due to insufficient knowledge or score."
        elif avg_score >= 8.5:
            perf_badge = '<span class="badge excellent">🌟 Excellent</span>'
            decision = "Select Candidate"
            decision_msg = "✅ AI strongly recommends selecting this candidate."
        elif avg_score >= 7:
            perf_badge = '<span class="badge good">⭐ Good</span>'
            decision = "Select Candidate"
            decision_msg = "✅ AI recommends selecting this candidate."
        elif avg_score >= 5:
            perf_badge = '<span class="badge average">Average</span>'
            decision = "Review Candidate"
            decision_msg = "⚠️ AI recommends reviewing further before making a selection."
        else:
            perf_badge = '<span class="badge poor">Needs Improvement</span>'
            decision = "Do NOT Select"
            decision_msg = "❌ AI recommends NOT selecting this candidate due to low performance or knowledge."
        if avg_know >= 8.5:
            know_badge = '<span class="badge excellent">🧠 Strong Knowledge</span>'
        elif avg_know >= 7:
            know_badge = '<span class="badge good">🧩 Good Knowledge</span>'
        elif avg_know >= 4:
            know_badge = '<span class="badge average">Basic Knowledge</span>'
        else:
            know_badge = '<span class="badge poor">Weak Knowledge</span>'

        with st.expander("Show Interview Score & AI Recommendation", expanded=True):
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:1.5em; margin-bottom:10px;">
                {perf_badge}
                {know_badge}
                <span style="font-size:1.18em; font-weight:700; color:#2e5cfc;">
                    Overall: {avg_score:.2f}/10<br>Knowledge: {avg_know:.2f}/10
                </span>
            </div>
            """, unsafe_allow_html=True)
            st.write(f"### AI Recommendation: {decision}")
            st.info(decision_msg)

    st.header("🗣️ Speaker Highlights & Feedback")
    for card in analysis_cards:
        st.subheader(f"{card['speaker']} | Score: {card['score']:.1f}/10 | Tone: {card['tone']}")
        st.write(f"Knowledge Score: {card['knowledge_score']:.1f}/10 | Keywords Found: {', '.join(card['knowledge_found']) or 'N/A'}")
        st.write(f"**Confidence:** {card['confidence']:.2f} | **Empathy**: {card['empathy']}")
        st.write(f"**Clarity:** {card['clarity']}")
        st.success(f"Summary: {card['summary']}")
        st.info(f"Keywords: {card['keywords']}")
        st.markdown("---")

    st.header("🔎 AI Summary & Professional Guidance")
    session_summary = summarizer(transcript, max_length=80, min_length=20, do_sample=False)[0]['summary_text']
    st.success(session_summary)

    pros, cons = [], []
    if avg_score > 7: pros.append("Demonstrates good confidence and communication skills.")
    if avg_know >= 7: pros.append("Shows solid knowledge in the selected domain/subdomain.")
    if any(card['empathy'] > 0 for card in analysis_cards): pros.append("Displays empathy or engaging responses.")
    if all(card['confidence'] > 0.6 for card in analysis_cards): pros.append("Consistently confident throughout session.")
    if avg_score < 7: cons.append("Needs to improve overall interview performance or communication.")
    if avg_know < 7: cons.append("Domain knowledge is below desired expectations.")
    if any(card['empathy'] == 0 for card in analysis_cards): cons.append("Empathy and engagement could be strengthened.")
    if any(card['knowledge_score'] < 7 for card in analysis_cards): cons.append("Did not utilize all key technical/domain vocabulary.")

    st.subheader("AI Feedback: Pros and Cons")
    col_pros, col_cons = st.columns(2)
    with col_pros:
        st.markdown("**Pros:**")
        for item in pros:
            st.success(f"✔ {item}")
    with col_cons:
        st.markdown("**Cons:**")
        for item in cons:
            st.error(f"✖ {item}")

    st.markdown("""
    <div style='margin-top:1.3em;background:#eef8fc;border-radius:1em;padding:1.1em;'>
        <b>Best Practices:</b>
        <ul>
            <li>Be concise, use examples & structure in every answer</li>
            <li>Demonstrate domain knowledge using key industry terms</li>
            <li>Practice for confidence and empathetic communication</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.balloons()

    with st.expander("🔮 Innovation: Let AI Join Teams/Zoom/Meet and Evaluate Live", expanded=False):
        st.markdown("""
        AI InterviewBot could join your meeting live, analyze in real time, and send feedback to chat or a dashboard!

        **How to Build:**
        - Use Teams/Zoom/Meet SDK to get audio/transcript.
        - Stream to MentorFlow backend.
        - Present results in chat or post-meeting summary.
        """)
else:
    st.info("Upload or paste transcript, then click Analyze Now.")

st.caption("MentorFlow | Professional AI Interview Analyzer | © 2025 Comet Assistant Team")
