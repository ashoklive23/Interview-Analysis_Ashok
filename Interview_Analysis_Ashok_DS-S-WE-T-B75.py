import streamlit as st
import requests
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from streamlit_lottie import st_lottie  # <--- Correct import

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

vader = SentimentIntensityAnalyzer()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

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
    st.markdown('<div style="font-size:1.9em;font-weight:bold;color:#2e5cfc;">MentorFlow</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.05em; color:#444;">AI Interview Analyzer</div>', unsafe_allow_html=True)
    st.markdown("---")
    industry = st.selectbox(
        "Interview Domain",
        ['IT', 'Supply Chain', 'Testing', 'BPO', 'Manufacturing', 'Banking', 'Healthcare', 'Retail', 'Marketing', 'Education', 'Other']
    )
    subdomain_options = {
        "IT": ["Python", "SQL", "Java", "Data Science", "Cloud", "Cybersecurity", "DevOps", "Frontend", "Backend"],
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

st.title("MentorFlow Interview Analytics")
st.write("**Paste your interview/group discussion transcript below and click Analyze.**")

text_input = st.text_area("Paste transcript here:", height=200)
analyze_btn = st.button("📝 Analyze Now", use_container_width=True)

if analyze_btn:
    transcript = text_input.strip()
    if not transcript or len(transcript) < 10:
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

else:
    st.info("Paste your transcript and click Analyze Now.")

st.caption("MentorFlow | Professional AI Interview Analyzer | © 2025 Comet Assistant Team")
