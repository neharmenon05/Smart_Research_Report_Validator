# app_final.py
import re
import json
import uuid
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfReader
import docx
from difflib import SequenceMatcher
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import base64

# ====================
# Config & constants
# ====================
st.set_page_config(page_title="Smart Research Report Validator — FLAT", layout="wide")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
sns.set_style("whitegrid")

# ====================
# Helper functions
# ====================
def roman(n: int) -> str:
    val = [1000,900,500,400,100,90,50,40,10,9,5,4,1]
    syms = ["M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"]
    res = ""
    i = 0
    while n > 0:
        for _ in range(n // val[i]):
            res += syms[i]
            n -= val[i]
        i += 1
    return res

def lines_from_text(text: str):
    return [(i+1, line.rstrip()) for i, line in enumerate(text.splitlines())]

def extract_text_from_file(uploaded_file):
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                pages.append(t)
        return "\n".join(pages)
    if name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs])
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    st.error("Unsupported file type.")
    return ""

SECTION_SYNONYMS = {
    "abstract": ["abstract","summary"],
    "introduction": ["introduction","background"],
    "related work": ["related work","literature review","prior work","state of the art"],
    "methodology": ["methodology","methods","approach","experimental setup"],
    "results": ["results","findings","experiments"],
    "discussion": ["discussion","analysis","interpretation"],
    "conclusion": ["conclusion","summary of findings","closing remarks","future work"],
    "references": ["references","bibliography","works cited"]
}
EXPECTED_SECTIONS = list(SECTION_SYNONYMS.keys())

def fuzzy_match(a: str, b: str, threshold=0.72):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

def extract_canonical_sections(doc: str):
    headings = []
    for ln, line in lines_from_text(doc):
        clean = line.strip()
        if len(clean.split()) > 8:  # skip very long lines
            continue
        norm = re.sub(r'^[IVXLCDM\d\.\-\s]*','', clean).strip("*_ ").lower()
        for canon, syns in SECTION_SYNONYMS.items():
            matched = False
            for syn in syns:
                if fuzzy_match(norm, syn):
                    headings.append((ln, canon, clean))
                    matched = True
                    break
            if matched:
                break
    # keep first occurrence per canonical section
    seen = set()
    out = []
    for ln, canon, orig in headings:
        if canon not in seen:
            out.append((ln, canon, orig))
            seen.add(canon)
    return out

def extract_section_texts(doc: str, headings):
    lines = dict(lines_from_text(doc))
    sorted_headings = sorted(headings, key=lambda x: x[0])
    out = {}
    maxln = max(lines.keys()) if lines else 0
    for i, (ln, canon, orig) in enumerate(sorted_headings):
        start = ln
        end = sorted_headings[i+1][0] - 1 if i+1 < len(sorted_headings) else maxln
        chunk = []
        for L in range(start+1, end+1):
            if L in lines:
                chunk.append(lines[L])
        out[canon] = {"heading": orig, "line": ln, "text": "\n".join(chunk).strip()}
    return out

def issue_dict(typ, line, issue, excerpt):
    return {"id": str(uuid.uuid4()), "type": typ, "line": line, "issue": issue, "excerpt": excerpt}

# ========== Checks ==========
def check_balanced_delimiters(doc: str):
    issues = []
    stack = []
    pairs = {"(":")","[":"]","{":"}","<":">"}
    for ln, text in lines_from_text(doc):
        for ch in text:
            if ch in pairs:
                stack.append((ch, pairs[ch], ln, text))
            elif ch in pairs.values():
                if not stack:
                    issues.append(issue_dict("delimiter", ln, f"Extra closing {ch}", text))
                else:
                    open_ch, expected, oln, ot = stack.pop()
                    if ch != expected:
                        issues.append(issue_dict("delimiter", ln, f"Mismatched {open_ch} closed with {ch}", text))
    for open_ch, expected, ln, text in stack:
        issues.append(issue_dict("delimiter", ln, f"Unclosed {open_ch}", text))
    return issues

def check_numbering(doc: str):
    issues = []
    expected = 1
    for ln, line in lines_from_text(doc):
        m = re.match(r'^(\d+)[\).]\s*', line.strip())
        if m:
            num = int(m.group(1))
            if num != expected:
                issues.append(issue_dict("numbering", ln, f"Numbering expected {expected}, found {num}", line))
            expected = num + 1
    return issues

def check_and_or(doc: str):
    issues = []
    for ln, line in lines_from_text(doc):
        low = line.lower()
        count_and = low.count(" and ")
        count_or = low.count(" or ")
        if "and/or" in low or "and / or" in low:
            issues.append(issue_dict("andor", ln, "Use of 'and/or' found — consider clarifying", line))
        elif count_and + count_or > 3:
            issues.append(issue_dict("andor", ln, f"Many conjunctions (and/or): {count_and + count_or} occurrences — consider simplifying", line))
    return issues

def check_parallelism_sections(section_texts):
    issues = []
    for canon, info in section_texts.items():
        text = info.get("text", "")
        if not text:
            continue
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        items = [l for l in lines if re.match(r'^(\-|\*|\d+\.)\s+', l)]
        if len(items) >= 3:
            starts = []
            for it in items:
                it2 = re.sub(r'^(\-|\*|\d+\.)\s+', '', it).strip()
                tok = it2.split()[0] if it2.split() else ""
                starts.append(tok.lower())
            ing_count = sum(1 for s in starts if s.endswith("ing"))
            base_count = len(starts) - ing_count
            if ing_count > 0 and base_count > 0:
                issues.append(issue_dict("parallelism", info.get("line", 0),
                    f"Parallelism in {canon}: list items mix '-ing' and base forms; make list items parallel.",
                    "\n".join(items[:6])))
    return issues

def compute_progress_from_issues(issues, total_tokens):
    err_tokens = sum(len(i.get("excerpt","").split()) for i in issues)
    ratio = err_tokens / max(1, total_tokens)
    return max(0.0, min(1.0, 1.0 - ratio))

# ========== Gemini call ==========
def call_gemini(prompt: str, api_key: str, timeout: int = 30) -> str:
    if not api_key:
        return "No API key provided."
    headers = {"Content-Type": "application/json"}
    url = f"{GEMINI_URL}?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                return json.dumps(data)
        else:
            return f"Request failed: {r.status_code} {r.text}"
    except Exception as e:
        return f"Request failed: {e}"

# ========== PDF generation ==========
def make_pdf_report(title, summary_bullets, suggestions, scores, issues):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, title)
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "AI Summary:")
    y -= 18
    c.setFont("Helvetica", 11)
    for b in summary_bullets:
        if y < margin+50:
            c.showPage(); y = height - margin
        c.drawString(margin+10, y, f"• {b}")
        y -= 14
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Enhancement Suggestions:")
    y -= 18
    c.setFont("Helvetica", 11)
    for s in suggestions:
        if y < margin+50:
            c.showPage(); y = height - margin
        c.drawString(margin+10, y, f"• {s}")
        y -= 14
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Scores:")
    y -= 18
    c.setFont("Helvetica", 11)
    for k,v in scores.items():
        c.drawString(margin+10, y, f"- {k.title()}: {v}")
        y -= 14
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Detected Issues:")
    y -= 16
    c.setFont("Helvetica", 10)
    for it in issues:
        if y < margin+50:
            c.showPage(); y = height - margin
        c.drawString(margin+5, y, f"- Line {it.get('line')}: {it.get('issue')}")
        y -= 12
        excerpt = it.get("excerpt","").strip().replace("\n"," ")
        if len(excerpt) > 200: excerpt = excerpt[:197] + "..."
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(margin+12, y, f"  Excerpt: {excerpt}")
        c.setFont("Helvetica", 10)
        y -= 16
    c.save()
    buffer.seek(0)
    return buffer

def pdf_download_link(buffer, filename="report.pdf"):
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download PDF report</a>'
    return href

# ==========================
# App: Upload at top
# ==========================
st.title("📘 Smart Research Report Validator — Formal Languages & Automata Theory")

api_key = st.sidebar.text_input("Gemini API Key", type="password", key="gemini_key_main")

uploaded_file = st.file_uploader("Upload your paper (PDF, DOCX, TXT)", type=["pdf","docx","txt"], key="uploader_main")
text_area_key = "paste_area_main"
if uploaded_file:
    doc_text = extract_text_from_file(uploaded_file)
    st.session_state[text_area_key] = doc_text
else:
    doc_text = st.text_area("Or paste your paper text here:", height=300, key=text_area_key)

if not doc_text or not doc_text.strip():
    st.info("Please upload or paste a document to analyze.")
    st.stop()

# ========== Recompute checks when doc changes ==========
doc_hash = hash(doc_text)
if st.session_state.get("last_doc_hash") != doc_hash:
    st.session_state["last_doc_hash"] = doc_hash
    # initialize containers
    st.session_state["issues"] = []
    st.session_state["accepted"] = []
    # run checks
    delim = check_balanced_delimiters(doc_text)
    numbering = check_numbering(doc_text)
    andor = check_and_or(doc_text)
    headings = extract_canonical_sections(doc_text)
    sections_texts = extract_section_texts(doc_text, headings)
    parallel = check_parallelism_sections(sections_texts)
    # Combine issues
    st.session_state["issues"] = delim + numbering + andor + parallel

# variables
issues = st.session_state.get("issues", [])
accepted = st.session_state.get("accepted", [])
total_tokens = max(1, len(doc_text.split()))

# ========== Tabs layout ==========
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Report Structural Issues", "Formatting Issues", "Charts", "AI Insights", "Project Concepts"])

# -------- Tab 1: Structural Issues --------
with tab1:
    st.header("Report Structural Issues — Sections, ordering & flow")
    headings = extract_canonical_sections(doc_text)
    section_texts = extract_section_texts(doc_text, headings)

    st.subheader("Detected sections (first occurrence)")
    if headings:
        for idx, (ln, canon, orig) in enumerate(headings, start=1):
            st.markdown(f"**{roman(idx)}. {canon.title()}** — line {ln}  \n> {orig}")
    else:
        st.info("No canonical section headings detected (heuristic).")

    st.subheader("Missing / Out-of-order sections")
    # check missing
    found_canons = [c for (_, c, _) in headings]
    missing = [s for s in EXPECTED_SECTIONS if s not in found_canons]
    if missing:
        st.warning("Missing sections detected:")
        for m in missing:
            st.markdown(f"- ❗ {m.title()}")
    else:
        st.success("All expected canonical sections present (best-effort).")

    # simple ordering check: compare order of EXPECTED_SECTIONS vs found_canons
    # If found_canons is a subsequence of EXPECTED_SECTIONS but differs in order, flag
    if found_canons:
        # compute order indexes
        expected_index = {name:i for i,name in enumerate(EXPECTED_SECTIONS)}
        found_indexes = [expected_index.get(c, -1) for c in found_canons]
        if any(x==-1 for x in found_indexes):
            st.info("Some detected headings are nonstandard and not in expected list.")
        else:
            if found_indexes != sorted(found_indexes):
                st.error("Section ordering appears inconsistent vs typical structure. Consider reordering sections.")
                # show where
                st.write("Detected order:", " → ".join(found_canons))
            else:
                st.success("Section order looks reasonable.")

    # show short flow hint using headings sequence
    st.subheader("Flow hint")
    if found_canons:
        st.write("Suggested reading flow based on detected sections:")
        st.write(" → ".join(found_canons))
    else:
        st.write("No flow can be inferred.")

# -------- Tab 2: Formatting Issues (with Accept/Ignore) --------
with tab2:
    st.header("Formatting Issues — delimiters, numbering, and/or, parallelism")
    if not issues:
        st.success("No formatting issues detected.")
    else:
        # show each issue as a card with Accept/Ignore buttons and excerpt
        for it in list(issues):  # copy so we can mutate
            typ = it.get("type","other")
            color = {
                "delimiter":"#f9c74f",
                "numbering":"#f94144",
                "andor":"#90be6d",
                "parallelism":"#577590"
            }.get(typ, "#adb5bd")
            cols = st.columns([7,1,1])
            with cols[0]:
                st.markdown(f"**Line {it.get('line','?')}** — {it.get('issue')}")
                st.code(it.get("excerpt","")[:800])
            accept_btn = cols[1].button("Accept", key=f"accept_{it['id']}")
            ignore_btn = cols[2].button("Ignore", key=f"ignore_{it['id']}")
            if accept_btn:
                st.session_state["accepted"].append(it)
                st.session_state["issues"] = [x for x in st.session_state["issues"] if x["id"] != it["id"]]
                st.experimental_rerun()
            if ignore_btn:
                st.session_state["issues"] = [x for x in st.session_state["issues"] if x["id"] != it["id"]]
                st.experimental_rerun()

    # progress
    remaining = st.session_state.get("issues", [])
    progress_val = compute_progress_from_issues(remaining, total_tokens)
    st.subheader("Progress")
    st.progress(progress_val)
    st.write(f"Progress metric (heuristic): {round(progress_val*100,2)}%")

# -------- Tab 3: Charts --------
with tab3:
    st.header("Charts & Scores")
    # error distribution bar
    counts = {"Delimiters":0, "Numbering":0, "Parallelism":0, "And/Or":0}
    for e in st.session_state.get("issues", []):
        t = e.get("type","")
        if t == "delimiter": counts["Delimiters"] += 1
        elif t == "numbering": counts["Numbering"] += 1
        elif t == "parallelism": counts["Parallelism"] += 1
        elif t == "andor": counts["And/Or"] += 1
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(list(counts.keys()), list(counts.values()), color=sns.color_palette("Set2"))
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # AI scores visualization if available (cached in session after AI run)
    ai_scores = st.session_state.get("ai_scores")
    if ai_scores:
        st.subheader("AI Scores Visualized")
        labels = ["Clarity","Content","Structure"]
        vals = [ai_scores.get("clarity",0), ai_scores.get("content",0), ai_scores.get("structure",0)]
        # radar / spider
        angles = [n / float(len(labels)) * 2 * 3.14159 for n in range(len(labels))]
        vals_closed = vals + vals[:1]
        angles_closed = angles + angles[:1]
        fig2 = plt.figure(figsize=(5,4))
        ax2 = fig2.add_subplot(111, polar=True)
        ax2.plot(angles_closed, vals_closed, 'o-', linewidth=2)
        ax2.fill(angles_closed, vals_closed, alpha=0.25)
        ax2.set_thetagrids([a*180/3.14159 for a in angles], labels)
        ax2.set_ylim(0,100)
        st.pyplot(fig2)
        st.metric("Cumulative Score", f"{round(sum(vals)/3,2)}/100")
    else:
        st.info("AI scores will appear on AI Insights tab after running the AI (enter API key and run).")

# -------- Tab 4: AI Insights (bullet-style) --------
with tab4:
    st.header("AI Insights — Reviewer-style bullets")
    if not api_key:
        st.info("Enter Gemini API key in the sidebar and press 'Generate AI Review' to get bullet-style summary, suggestions and scores.")
    else:
        if st.button("Generate AI Review", key="gen_ai_btn"):
            PROMPT_CHARS = 3600
            trimmed = doc_text[:PROMPT_CHARS]
            prompt = (
                "You are an expert academic reviewer. For the text below:\n\n"
                "1) Provide a concise bullet-style summary (5 bullets) of the paper.\n"
                "2) Provide up to 6 concrete enhancement suggestions (bullet-style).\n"
                "3) Output a JSON object with numeric scores (0-100) for clarity, content, and structure. PLACE THE JSON OBJECT ON A SEPARATE LINE AT THE END.\n\n"
                "Paper content:\n" + trimmed
            )
            with st.spinner("Calling Gemini (trimmed input)..."):
                resp = call_gemini(prompt, api_key, timeout=40)
            if resp.startswith("Request failed") or resp == "No API key provided.":
                st.error(resp)
            else:
                # parse bullets for summary
                bullets = re.findall(r'^(?:\s*[-\*\u2022]|\s*\d+\.)\s*(.+)', resp, flags=re.MULTILINE)
                if not bullets:
                    lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
                    bullets = lines[:5]
                st.subheader("AI Summary (5 bullets)")
                for b in bullets[:5]:
                    st.write("•", b)

                # suggestions (take next short lines after bullets or look for keywords)
                suggestions = []
                # locate a block with 'suggest' or similar
                m = re.search(r'(?i)(suggest|improv|enhanc).{0,50}[:\-\n]\s*(.+?)(?:\n\n|\n\s*\{)', resp, flags=re.DOTALL)
                if m:
                    raw = m.group(2)
                    suggestions = re.findall(r'^(?:\s*[-\*\u2022]|\s*\d+\.)\s*(.+)', raw, flags=re.MULTILINE)
                if not suggestions:
                    # fallback: take some lines not used in summary
                    lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
                    suggestions = lines[5:11]
                if suggestions:
                    st.subheader("Enhancement Suggestions")
                    for s in suggestions[:6]:
                        st.write("•", s)

                # parse JSON scores
                scores = {"clarity":0,"content":0,"structure":0}
                jm = re.search(r'(\{.*\})', resp, flags=re.DOTALL)
                if jm:
                    try:
                        parsed = json.loads(jm.group(1))
                        for k in ["clarity","content","structure"]:
                            if k in parsed:
                                try:
                                    scores[k] = float(parsed[k])
                                except:
                                    try: scores[k] = float(str(parsed[k]).strip())
                                    except: scores[k] = 0.0
                    except Exception:
                        pass
                st.session_state["ai_scores"] = scores

                st.subheader("AI Scores (parsed)")
                st.write(scores)
                st.success("AI review generated and scores stored (view Charts tab for visualizations).")

# -------- Tab 5: Project Concepts (FLAT) --------
with tab5:
    st.header("Project Concepts — Formal Languages & Automata Theory")
    st.markdown("""
This validator intentionally uses automata-inspired techniques:

- **Balanced delimiters** → *Pushdown Automaton (PDA)*: matching nested parentheses/brackets is a classical context-free problem (use of stack).
- **Numbering & regex scans** → *Finite Automata (FA)* / Regular Expressions: simple token scanning and pattern recognition.
- **Section membership & order** → *Grammar-checking / membership*: we attempt to match a document against an expected 'paper language' template.
- **Parallelism & conjunction checks** → heuristic, surface-level syntactic checks (regular scans), which approximate well-formedness properties.
- **AI (Gemini)** acts as an external oracle (reviewer) to score and summarize content beyond syntactic checks.

Include these mappings in your project report to show academic grounding.
""")

# ========== Download PDF report ==========
st.markdown("---")
st.header("Download Evaluation Report")
if st.button("Generate PDF report (includes AI, issues, scores)", key="gen_pdf"):
    # prepare data for PDF
    ai_scores = st.session_state.get("ai_scores", {"clarity":0,"content":0,"structure":0})
    # attempt to gather summary & suggestions from latest AI response if present
    # fallback: use placeholders
    summary_bullets = []
    suggestions = []
    # try to get them from previous AI output - we don't store full text, so ask user to generate if missing
    if "ai_scores" not in st.session_state:
        st.warning("No AI run available. Generate AI review on the AI Insights tab first.")
    else:
        # Very simple summary: use small placeholders (detailed summary comes from AI run displayed earlier)
        # You may enhance by saving the full AI response into session_state during generation
        summary_bullets = ["AI summary produced in the AI Insights tab."]  # placeholder
        suggestions = ["AI suggestions produced in the AI Insights tab."]  # placeholder

        buffer = make_pdf_report("Smart Research Report Validator — Evaluation", summary_bullets, suggestions, ai_scores, st.session_state.get("accepted", []) + st.session_state.get("issues", []))
        st.markdown(pdf_download_link(buffer, "evaluation_report.pdf"), unsafe_allow_html=True)

st.caption("Tip: run AI Insights first (Generate AI Review) to include AI summary & scores in the PDF.")

# Footer
st.markdown("---")
