import streamlit as st
import requests
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_lottie import st_lottie

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    from gensim.summarization import summarize
    summarizer_available = True
except ImportError:
    summarizer_available = False

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
        ['IT', 'Supply Chain', 'Testing', 'BPO', 'Manufacturing', 'Banking', 'Healthcare',
         'Retail', 'Marketing', 'Education', 'Other']
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
        "Subdomain/Role", subdomain_options.get(industry, ["General"])
    )
    exp_level = st.radio("Candidate Type", ["Fresher", "Experienced"], index=0)
    round_type = st.selectbox("Round Type", ['Interview', 'Group Discussion', 'Mock'])

st.title("MentorFlow Interview Analytics")
st.write("Paste your transcript below and click **Analyze Now**.")

# THE ONLY INPUT WIDGET IS NEXT LINE
text_input = st.text_area("Paste interview/group transcript here:", height=200)
analyze_btn = st.button("📝 Analyze Now", use_container_width=True)

if analyze_btn:
    transcript = text_input.strip()
    if not transcript or len(transcript) < 10:
        st.warning("Insufficient data for analysis. Please paste a transcript.")
        st.stop()

    # ... (rest of the analysis code as before; NO file upload logic anywhere)
