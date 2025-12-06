import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

API_KEY = "AIzaSyDlnSBUgoN2m94xmaFY2WIT-GjYC8MOUUg"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"


# ---------------- GEMINI CALL (FIXED PARSING) ---------------- #
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


def query_api(text):
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": f"""Classify the following news as REAL or FAKE.
Strict answer format:
1st line → REAL or FAKE
2nd line → Short explanation

Text:
{text}"""}]}
        ]
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        raw_text = safe_extract_text(result)

        lines = raw_text.split("\n", 1)
        classification = lines[0].strip().upper() if lines else "UNSURE"
        explanation = lines[1].strip() if len(lines) > 1 else "No explanation."

        if "REAL" in classification:
            return "REAL", explanation
        elif "FAKE" in classification:
            return "FAKE", explanation
        else:
            return "UNSURE", explanation

    except Exception as e:
        return f"ERROR: {e}", "Explanation not available due to error."


def get_true_info(fake_text):
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": f"""The following statement is FAKE.
Provide the correct factual version in 1–2 sentences.

Fake statement:
{fake_text}"""}]}
        ]
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        return safe_extract_text(result)
    except:
        return "No correction available."


# ---------------- BERT & ROBERTA ---------------- #

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


# ---------------- UTILITIES ---------------- #

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
    text = clean_text(text)
    if url:
        source_name = get_source_name(url)
        if source_name:
            return "REAL", f"This article is from a trusted and reputable source: **{source_name}**."

    return query_api(text)


# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="Fake News Detection App", page_icon="📰", layout="wide")

st.title("📰 Fake News Detection")

input_type = st.radio("Choose Input Type", ["Text", "URL"])
user_input = ""
page_url = ""

if input_type == "Text":
    user_input = st.text_area("Enter news text")

else:
    page_url = st.text_input("Enter URL")
    if page_url:
        scraped = scrape_url(page_url)
        if scraped:
            st.text_area("Extracted Article", scraped, height=300)
            user_input = scraped
        else:
            st.warning("Could not scrape URL.")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter text or URL.")
    else:
        result, explanation = final_decision(user_input, page_url)

        if result == "REAL":
            st.success("🟢 REAL NEWS")
        elif result == "FAKE":
            st.error("🔴 FAKE NEWS")
            st.info("Correct Information:\n" + get_true_info(user_input))
        else:
            st.warning("⚠ UNSURE")

        st.info("Why:\n" + explanation)
