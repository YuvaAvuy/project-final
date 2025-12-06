import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ---------- CONFIG ----------
API_KEY = "AIzaSyA0NTeHJveTXalBlqJ1AWx8OIn7AIgiJJA"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# ---------- GEMINI HELPERS (robust parsing) ----------
def safe_extract_text(result):
    """Extract Gemini text safely regardless of JSON structure."""
    try:
        candidate = result.get("candidates", [{}])[0]
        content = candidate.get("content", {})
        if isinstance(content, list):
            content = content[0]
        parts = content.get("parts", [{}])
        return parts[0].get("text", "").strip()
    except:
        return ""


def call_gemini(prompt, timeout=30):
    """Single helper to call Gemini with a prompt and return text."""
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(API_URL, headers=headers, json=data, timeout=timeout)
        resp.raise_for_status()
        return safe_extract_text(resp.json())
    except Exception as e:
        return f"ERROR_CALLING_GEMINI: {e}"


# ---------- CORE API QUERIES ----------
def query_api_classify(text):
    """Ask Gemini to classify the news as REAL or FAKE and give an explanation."""
    prompt = f"""Classify the following news as REAL or FAKE.
Answer format:
1st line -> REAL or FAKE
2nd+ lines -> A clear explanation (2-4 sentences) why you classified it that way (credibility, language patterns, claims, sources).
Text:
{text}"""
    raw = call_gemini(prompt)
    # parse
    lines = raw.split("\n", 1)
    classification = lines[0].strip().upper() if lines else "UNSURE"
    explanation = lines[1].strip() if len(lines) > 1 else (raw if raw else "No explanation provided.")
    if "REAL" in classification:
        return "REAL", explanation
    elif "FAKE" in classification:
        return "FAKE", explanation
    elif raw.startswith("ERROR_CALLING_GEMINI"):
        return raw, "Explanation not available due to API error."
    else:
        return "UNSURE", explanation


def query_api_simple_explain(text, classification):
    """Ask Gemini to produce a simple, 3-bullet explanation for non-technical users."""
    prompt = f"""You answered '{classification}' for the text below.
Now write a short, very simple explanation that an unskilled person can understand.
- Start with one sentence summary.
- Then give exactly 3 short bullet points (one phrase each) that explain WHY (e.g., 'no reliable source', 'sensational language', 'confirmed by official site').
- Keep each bullet under 8 words.

Text:
{text}"""
    return call_gemini(prompt)


def query_api_detailed_explain(text, classification):
    """Ask Gemini for a more detailed explanation (longer, stepwise) to show evidence and sources."""
    prompt = f"""You marked the news as '{classification}'.
Provide a detailed explanation (3-5 short paragraphs) that:
- Points to specific issues or supporting facts,
- Mentions whether sources exist or not,
- Suggests what a reader should check to verify (give 3 concrete checks).

Text:
{text}"""
    return call_gemini(prompt)


def get_true_info(fake_text):
    """Ask Gemini to give the factual correction for a fake statement."""
    prompt = f"""The following statement is FAKE.
Provide the correct factual version in 1–2 clear sentences and mention a reliable source type (e.g., 'official govt statement', 'Reuters').
Fake statement:
{fake_text}"""
    return call_gemini(prompt)


# ---------- LOCAL MODEL SIGNALS (BERT & ROBERTA) ----------
@st.cache_resource
def load_bert_model():
    model = AutoModelForSequenceClassification.from_pretrained("omykhailiv/bert-fake-news-recognition")
    tokenizer = AutoTokenizer.from_pretrained("omykhailiv/bert-fake-news-recognition")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)


@st.cache_resource
def load_roberta_model():
    return pipeline("zero-shot-classification", model="roberta-large-mnli")


bert_pipeline = load_bert_model()
roberta_pipeline = load_roberta_model()


def local_model_signals(text):
    """Return short signals from local models to support the Gemini verdict."""
    # Limit text length for local models for speed
    short_text = text[:512]
    try:
        bert_res = bert_pipeline(short_text, truncation=True)[0]  # {'label': 'LABEL_1', 'score': 0.9}
        # Model's labels depend on checkpoint; try to map common patterns
        label = bert_res.get("label", "")
        score = float(bert_res.get("score", 0.0))
        # Normalize label: many fake-news models use 'REAL'/'FAKE' or 'LABEL_0' / 'LABEL_1'
        if label.upper() in ("REAL", "TRUE"):
            bert_label = "REAL"
        elif label.upper() in ("FAKE", "FALSE"):
            bert_label = "FAKE"
        else:
            # fallback mapping by inspecting label string for 'REAL' or 'FAKE'
            bert_label = "REAL" if "REAL" in label.upper() or "TRUE" in label.upper() else "FAKE"
    except Exception as e:
        bert_label = "ERROR"
        score = 0.0

    try:
        roberta_res = roberta_pipeline(short_text, candidate_labels=["REAL", "FAKE"], truncation=True)
        # returns {'labels': [...], 'scores': [...]}
        rn_label = roberta_res["labels"][0]
        rn_score = float(roberta_res["scores"][0])
    except Exception as e:
        rn_label = "ERROR"
        rn_score = 0.0

    return {
        "bert_label": bert_label,
        "bert_score": round(score, 3),
        "roberta_label": rn_label,
        "roberta_score": round(rn_score, 3),
    }


# ---------- UTILITIES ----------
def clean_text(text):
    text = re.sub(r"\b\d{1,2}\s*(hours|minutes|ago)\b", "", text)
    text = re.sub(r"(share|save|click here|more details|read more)", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def scrape_url(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.title.string if soup.title else ""
        article_div = (
            soup.find("article")
            or soup.find("div", {"class": "articlebodycontent"})
            or soup.find("div", {"id": "content-body"})
        )

        if article_div:
            chunks = [elem.get_text().strip() for elem in article_div.find_all(["p", "li"]) if len(elem.get_text().split()) > 5]
        else:
            chunks = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().split()) > 5]

        text = " ".join(chunks) or soup.get_text()
        return clean_text((title + "\n\n" + text)[:4000])
    except:
        return None


trusted_sources = {
    "thehindu.com": "The Hindu",
    "timesofindia.com": "Times of India",
    "hindustantimes.com": "Hindustan Times",
    "ndtv.com": "NDTV",
    "bbc.com": "BBC",
    "cnn.com": "CNN",
    "reuters.com": "Reuters",
    "apnews.com": "Associated Press",
}


def get_source_name(url):
    url_lower = url.lower()
    for domain, name in trusted_sources.items():
        if domain in url_lower:
            return name
    return None


def final_decision(text, url=""):
    """If trusted source is detected, short-circuit to REAL; otherwise ask Gemini."""
    text = clean_text(text)
    if url:
        source_name = get_source_name(url)
        if source_name:
            # still gather explanations but mark as REAL initially
            simple = f"This article is from a trusted source: {source_name}."
            # Ask Gemini to explain in simple language why this source makes it reliable
            simple_expl = query_api_simple_explain(text, "REAL")
            detailed_expl = query_api_detailed_explain(text, "REAL")
            return "REAL", simple, simple_expl, detailed_expl
    # else use classifier
    cls, expl = query_api_classify(text)
    simple_expl = query_api_simple_explain(text, cls) if not cls.startswith("ERROR") else "No simple explanation (API error)."
    detailed_expl = query_api_detailed_explain(text, cls) if not cls.startswith("ERROR") else "No detailed explanation (API error)."
    return cls, expl, simple_expl, detailed_expl


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
st.markdown("<h1 style='text-align:center'>📰 Fake News Detection</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    input_type = st.radio("Choose input type", ["Text", "URL"])
    user_input = ""
    page_url = ""
    if input_type == "Text":
        user_input = st.text_area("Enter news text here", height=220, placeholder="Paste or type the news content...")
    else:
        page_url = st.text_input("Enter article URL", placeholder="https://example.com/news-article")
        if page_url:
            scraped = scrape_url(page_url)
            if scraped:
                st.text_area("Extracted Article (from URL)", scraped, height=300)
                user_input = scraped
            else:
                st.warning("Could not scrape the URL. You can paste the article text manually.")

    analyze_btn = st.button("Analyze")

with col2:
    st.markdown("### Support Signals")
    st.markdown("- BERT & RoBERTa provide extra evidence.")
    st.markdown("- Gemini provides final human-level explanation.")

if analyze_btn:
    if not user_input or not user_input.strip():
        st.warning("Please enter text or provide a URL.")
    else:
        # Show extracted text always in an expander
        with st.expander("🔎 Extracted / Input Text (click to expand)"):
            st.write(user_input)

        # Local model signals
        with st.spinner("Running local model checks..."):
            signals = local_model_signals(user_input)

        # Final decision + explanations (calls Gemini)
        with st.spinner("Querying explanation model (Gemini) — this may take a few seconds..."):
            result, explanation, simple_expl, detailed_expl = final_decision(user_input, page_url)

        # Verdict
        if isinstance(result, str) and result.startswith("ERROR_CALLING_GEMINI"):
            st.error("⚠ Error calling Gemini API. Check your API key, billing, and permissions.")
            st.error(result)
        else:
            if result == "REAL":
                st.success("🟢 FINAL VERDICT: REAL")
            elif result == "FAKE":
                st.error("🔴 FINAL VERDICT: FAKE")
            elif result == "UNSURE":
                st.warning("⚠ FINAL VERDICT: UNSURE")
            else:
                # in case result contains API error text
                st.warning(f"⚠ FINAL VERDICT: {result}")

            # Human-friendly simple explanation (short, 3 bullets)
            st.markdown("## 🧾 Simple Explanation (for everyone)")
            if simple_expl and simple_expl.startswith("ERROR_CALLING_GEMINI"):
                st.info("No simple explanation available due to API error.")
            else:
                st.info(simple_expl)

            # Detailed explanation
            st.markdown("## 🔍 Detailed Explanation & Evidence")
            if explanation and explanation.startswith("ERROR_CALLING_GEMINI"):
                st.info("No detailed explanation available due to API error.")
            else:
                st.write(explanation)

            # Show longer structured detailed explanation (if available)
            st.markdown("### More detailed guidance")
            if detailed_expl and not detailed_expl.startswith("ERROR_CALLING_GEMINI"):
                st.write(detailed_expl)
            else:
                st.info("No additional detailed explanation available.")

            # If fake, provide corrected factual info
            if result == "FAKE":
                st.markdown("## ✅ Correct / Factual Information (if available)")
                correction = get_true_info(user_input)
                if correction and not correction.startswith("ERROR_CALLING_GEMINI"):
                    st.info(correction)
                else:
                    st.info("No correction available (API error or insufficient info).")

            # Show local model signals as supporting evidence
            st.markdown("## 🧪 Local Model Signals (supporting evidence)")
            st.write(f"- BERT model: {signals['bert_label']} (score {signals['bert_score']})")
            st.write(f"- RoBERTa (zero-shot): {signals['roberta_label']} (score {signals['roberta_score']})")

            # Extra tips for users (actionable verification steps)
            st.markdown("## ✅ How to verify this yourself (3 quick checks)")
            st.markdown("1. Search the headline on reputable news sites (Reuters, BBC, AP).")
            st.markdown("2. Check for an official source (govt website, official org).")
            st.markdown("3. Look for multiple reliable outlets reporting the same claim.")

# end
