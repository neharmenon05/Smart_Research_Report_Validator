# app.py
# =========================================================
# Smart Research Report Validator (PDA-based) — Streamlit
# =========================================================
# Features:
# - Balanced delimiters & quotes ((), {}, [], [], "", ''), citations [1]
# - Section order/coverage (Abstract → Introduction → Methods → Results → Discussion → Conclusion → References)
# - Numbering hierarchy (1, 1.1, 1.1.1 ...)
# - Symmetry in lists/conjunctions (parallel phrasing around "and"/"or")
# - CFG→PDA conformance for simplified research-paper structure
#
# Run:
#   pip install streamlit pypdf regex
#   streamlit run app.py
# =========================================================

import io
import json
import re
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import streamlit as st
from pypdf import PdfReader

# ----------------------- Streamlit UI -----------------------
st.set_page_config(page_title="Smart Research Report Validator (PDA)", page_icon="📘", layout="wide")

st.title("📘 Smart Research Report Validator (PDA-based)")
st.caption("Upload a research report PDF (or paste text) to validate structure using Pushdown Automata ideas.")

with st.sidebar:
    st.header("Input")
    pdf_file = st.file_uploader("Upload Research Report (PDF)", type=["pdf"])
    st.markdown("**Or paste text**")
    text_input = st.text_area("Paste report text here", height=180, placeholder="Paste raw text if no PDF...")
    run_btn = st.button("▶️ Analyze")

# ----------------------- Utilities -----------------------
def extract_text_from_pdf(file: io.BytesIO) -> str:
    """Extract text from PDF using pypdf."""
    reader = PdfReader(file)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(parts)

def normalize_text(doc: str) -> str:
    doc = doc.replace("\u00a0", " ")
    doc = re.sub(r"[ \t]+", " ", doc)
    doc = re.sub(r"\r\n?", "\n", doc)
    return doc.strip()

def split_sentences(doc: str) -> List[str]:
    # lightweight sentence splitter (periods, semicolons, question/exclamation)
    sents = re.split(r"(?<=[\.\?!;])\s+", doc)
    return [s for s in sents if s.strip()]

def tokenize(doc: str) -> List[str]:
    # words + punctuation tokens
    return re.findall(r"[A-Za-z]+|[\(\)\{\}\[\]<>“”\"'«»;:.,$]", doc)

# ----------------------- PDA Core -----------------------
class PDA:
    def __init__(self):
        self.stack = []
        self.state = 'q0'
        self.transitions = {}   # (state, input_char, stack_top) -> [(new_state, [ops])]
        self.accept_state = 'accept'

    def add_transition(self, state, input_char, stack_top, new_state, stack_ops):
        key = (state, input_char, stack_top)
        self.transitions.setdefault(key, []).append((new_state, stack_ops))

    def reset(self):
        self.stack = []
        self.state = 'q0'

def apply_ops(stack, ops):
    for op in ops:
        if op == 'pop' and stack:
            stack.pop()
        elif op.startswith('push:'):
            stack.append(op.split(':', 1)[1])

def cfg_to_pda(productions: List[Tuple[str, str]]) -> PDA:
    """
    productions: list of (LHS, RHS) where nonterminals are uppercase letters, terminals are single lowercase tokens.
    Use 'ε' for epsilon.
    """
    pda = PDA()
    pda.add_transition('q0', None, None, 'q1', ['push:S'])

    # Expand variables
    for lhs, rhs in productions:
        if rhs == 'ε':
            pda.add_transition('q1', None, lhs, 'q1', ['pop'])
        else:
            ops = ['pop'] + [f'push:{c}' for c in reversed(rhs)]
            pda.add_transition('q1', None, lhs, 'q1', ops)

    # Terminal consumption (single-char lowercase terminals)
    terms = set()
    for _, rhs in productions:
        for c in rhs:
            if c.islower():
                terms.add(c)
    for t in terms:
        pda.add_transition('q1', t, t, 'q1', ['pop'])

    pda.add_transition('q1', None, None, 'accept', [])
    return pda

def run_cfg_pda(pda: PDA, seq: List[str]) -> Tuple[bool, List[Tuple[str, Tuple[str, ...], int]]]:
    """
    seq: a list of single-character terminal symbols (strings of length 1).
    """
    start = ('q1', tuple(['S']), 0, [])
    q = deque([start])
    visited = set()

    while q:
        state, stack, i, hist = q.popleft()
        stack_top = stack[-1] if stack else None
        hist = hist + [(state, stack, i)]

        if state == 'accept' and not stack and i == len(seq):
            return True, hist

        config_id = (state, stack, i)
        if config_id in visited:
            continue
        visited.add(config_id)

        current_input = seq[i] if i < len(seq) else None

        # consume input
        for (st, sym, top), nexts in pda.transitions.items():
            if st == state and sym == current_input and (top == stack_top or top is None):
                for ns, ops in nexts:
                    new_stack = list(stack)
                    apply_ops(new_stack, ops)
                    q.append((ns, tuple(new_stack), i+1, hist))

        # epsilon moves
        for (st, sym, top), nexts in pda.transitions.items():
            if st == state and sym is None and (top == stack_top or top is None):
                for ns, ops in nexts:
                    new_stack = list(stack)
                    apply_ops(new_stack, ops)
                    q.append((ns, tuple(new_stack), i, hist))

    return False, hist

# ----------------------- Check 1: Balanced Delimiters & Quotes -----------------------
class BalanceChecker:
    OPEN = {'(' : ')', '{' : '}', '[' : ']', '<' : '>'}
    CLOSE = {')':'(', '}':'{', ']':'[', '>':'<'}

    def check(self, tokens: List[str]) -> Tuple[bool, List[str]]:
        issues = []
        stack = []
        # quote toggles
        quote_stack = []
        for idx, tok in enumerate(tokens):
            # math $...$ pair (toggle)
            if tok == '$':
                if quote_stack and quote_stack[-1] == '$':
                    quote_stack.pop()
                else:
                    quote_stack.append('$')
                continue

            # smart quotes
            if tok in ('“','”'):
                if tok == '“':
                    quote_stack.append('“')
                else:
                    if not quote_stack or quote_stack[-1] != '“':
                        issues.append(f"Mismatched ” at token {idx}")
                    else:
                        quote_stack.pop()
                continue

            if tok in ("'", '"'):
                if quote_stack and quote_stack[-1] == tok:
                    quote_stack.pop()
                else:
                    quote_stack.append(tok)
                continue

            if tok in self.OPEN:
                stack.append(tok)
            elif tok in self.CLOSE:
                if not stack or stack[-1] != self.CLOSE[tok]:
                    issues.append(f"Mismatched {tok} at token {idx}")
                else:
                    stack.pop()

        if stack:
            issues.append(f"Unclosed delimiters on stack: {stack}")
        if quote_stack:
            issues.append(f"Unclosed quotes/math: {quote_stack}")
        return (len(issues) == 0), issues

# ----------------------- Check 2: Section Order / Coverage -----------------------
SECTION_PATTERNS = {
    'a': re.compile(r'^\s*(abstract)\b', re.I),
    'i': re.compile(r'^\s*(introduction|background|overview|motivation|related work)\b', re.I),
    'm': re.compile(r'^\s*(methods?|materials and methods|methodology)\b', re.I),
    'r': re.compile(r'^\s*(results?|experiments?)\b', re.I),
    'd': re.compile(r'^\s*(discussion|analysis)\b', re.I),
    'c': re.compile(r'^\s*(conclusion|conclusions|concluding remarks)\b', re.I),
    'f': re.compile(r'^\s*(references|bibliography|works cited)\b', re.I),
    # Title is optional, often first line, we won’t strictly detect 't'
}

def extract_section_sequence(doc: str) -> List[str]:
    """Scan line starts for common section headings and produce terminal sequence."""
    seq = []
    for line in doc.splitlines():
        L = line.strip()
        if not L:
            continue
        # accept numbered headings like "1. Introduction"
        L2 = re.sub(r"^\d+(\.\d+)*[.)]\s*", "", L)
        captured = False
        for key, rx in SECTION_PATTERNS.items():
            if rx.match(L) or rx.match(L2):
                # avoid duplicates when same header repeats in wrapped lines
                if not seq or seq[-1] != key:
                    seq.append(key)
                captured = True
                break
        # ignore non-heading lines
    return seq

@dataclass
class SectionOrderResult:
    ok: bool
    issues: List[str]
    found: List[str]  # raw terminals in order

def check_section_order(doc: str) -> SectionOrderResult:
    """Strict order expected: a → i → m → r → d → c → f (a and d optional in some venues, but we keep them here)."""
    found = extract_section_sequence(doc)
    issues = []

    # enforce monotonic order using indices
    order = ['a', 'i', 'm', 'r', 'd', 'c', 'f']
    pos = {k: i for i, k in enumerate(order)}
    last = -1
    for k in found:
        if pos.get(k, -1) < last:
            issues.append(f"Section '{k}' appears out of order.")
        last = max(last, pos.get(k, -1))

    # coverage: require at least Abstract, Introduction, Methods, Results, Conclusion, References
    required = ['a', 'i', 'm', 'r', 'c', 'f']
    missing = [x for x in required if x not in found]
    if missing:
        issues.append(f"Missing required sections: {missing}")

    return SectionOrderResult(ok=(len(issues) == 0), issues=issues, found=found)

# ----------------------- Check 3: Numbering Hierarchy -----------------------
class NumberingChecker:
    def check(self, doc: str) -> Tuple[bool, List[str]]:
        issues = []
        lines = [l.rstrip() for l in doc.splitlines()]
        current = []

        def parse_head(s):
            m = re.match(r'^\s*(\d+(?:\.\d+)*)(?:[.)])?\s+', s)
            if m:
                return [int(x) for x in m.group(1).split('.')]
            return None

        for idx, line in enumerate(lines, 1):
            head = parse_head(line)
            if not head:
                continue
            if not current:
                if head != [1]:
                    issues.append(f"First top-level heading should be 1 at line {idx}")
                current = head[:]
                continue

            if len(head) == len(current):
                expected = current[:-1] + [current[-1] + 1]
                if head != expected:
                    issues.append(f"Expected {'.'.join(map(str,expected))} at line {idx}, found {'.'.join(map(str,head))}")
                current = head[:]
            elif len(head) == len(current) + 1:
                expected = current + [1]
                if head != expected:
                    issues.append(f"Expected {'.'.join(map(str,expected))} at line {idx}, found {'.'.join(map(str,head))}")
                current = head[:]
            elif len(head) < len(current):
                expected = current[:len(head)]
                expected[-1] += 1
                if head != expected:
                    issues.append(f"Expected {'.'.join(map(str,expected))} at line {idx}, found {'.'.join(map(str,head))}")
                current = head[:]
            else:
                issues.append(f"Jumped more than one level deeper at line {idx}: {'.'.join(map(str,head))}")

        return (len(issues) == 0), issues

# ----------------------- Check 4: Symmetry / Parallelism -----------------------
class SymmetryChecker:
    def check_sentence(self, sentence: str) -> Tuple[bool, List[str]]:
        # Check parallelism around first 'and'/'or'
        m = re.search(r'\b(and|or)\b', sentence, flags=re.I)
        if not m:
            return True, []
        conj = m.group(1).lower()
        left = sentence[:m.start()]
        right = sentence[m.end():]

        def norm(s):
            s = s.lower()
            s = re.sub(r'\b(shall|will|must|should|can|could|would|may|might)\b', 'shall', s)
            s = re.sub(r'[^a-z0-9 ]+', ' ', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return s.split()

        L = norm(left)
        R = norm(right)

        # stack-based mirror (loose): pop matches, ignore 'shall' repetition
        stack = []
        for w in L:
            stack.append(w)
        mismatches = []
        for w in R:
            if stack and stack[-1] == w:
                stack.pop()
            elif w == 'shall':
                continue
            else:
                mismatches.append(w)

        if stack and not all(x == 'shall' for x in stack):
            mismatches.extend([x for x in stack if x != 'shall'])

        ok = len(mismatches) == 0
        issues = []
        if not ok:
            issues.append(f"Asymmetry around '{conj}': {mismatches}")
        return ok, issues

    def check_document(self, doc: str) -> Tuple[bool, List[str]]:
        issues = []
        ok_all = True
        for sent in split_sentences(doc):
            ok, iss = self.check_sentence(sent)
            if not ok:
                ok_all = False
                issues.extend([f"[{sent.strip()}] -> {msg}" for msg in iss])
        return ok_all, issues

# ----------------------- Check 5: CFG→PDA for Research Skeleton -----------------------
class ResearchCFGChecker:
    """
    Terminals (single-char):
      a = Abstract
      i = Introduction (or Background/Related Work)
      m = Methods
      r = Results
      d = Discussion
      c = Conclusion(s)
      f = References
    Grammar (one possible strict path; illustrative):
      S -> A B F
      A -> a
      B -> i m r d c
    We allow a looser variant by providing ε-alternatives where needed if you want to relax.
    """
    def __init__(self, strict: bool = True):
        if strict:
            prods = [
                ('S', 'ABF'),
                ('A', 'a'),
                ('B', 'imrdc'),
                ('F', 'f'),
            ]
        else:
            # relaxed: allow Discussion optional
            prods = [
                ('S', 'ABF'),
                ('A', 'a'),
                ('B', 'imrc'),
                ('B', 'imrdc'),
                ('F', 'f'),
            ]
        self.pda = cfg_to_pda(prods)

    def check(self, doc: str) -> Tuple[bool, List[str]]:
        seq = extract_section_sequence(doc)
        # map list of symbols (strings like 'a','i',...) to terminals
        # Only keep terminals that appear in our grammar set
        valid = [x for x in seq if x in {'a','i','m','r','d','c','f'}]
        ok, _hist = run_cfg_pda(self.pda, valid)
        issues = []
        if not ok:
            issues.append("Document does not match the simplified research-paper CFG (Abstract→Introduction→Methods→Results→Discussion→Conclusion→References).")
        return ok, issues

# ----------------------- Orchestrator -----------------------
class ResearchValidator:
    def __init__(self):
        self.balance = BalanceChecker()
        self.order = None
        self.numbering = NumberingChecker()
        self.symmetry = SymmetryChecker()
        self.cfg = ResearchCFGChecker(strict=True)

    def validate(self, text: str) -> Dict[str, Any]:
        report = {
            "balanced_delimiters": {"ok": True, "issues": []},
            "section_order": {"ok": True, "issues": [], "found": []},
            "numbering": {"ok": True, "issues": []},
            "symmetry": {"ok": True, "issues": []},
            "cfg_research": {"ok": True, "issues": []},
        }

        doc = normalize_text(text)
        tokens = tokenize(doc)

        # 1) delimiters & quotes
        ok, issues = self.balance.check(tokens)
        report["balanced_delimiters"]["ok"] = ok
        report["balanced_delimiters"]["issues"] = issues

        # 2) section order
        so = check_section_order(text)
        report["section_order"]["ok"] = so.ok
        report["section_order"]["issues"] = so.issues
        report["section_order"]["found"] = so.found

        # 3) numbering
        ok, issues = self.numbering.check(text)
        report["numbering"]["ok"] = ok
        report["numbering"]["issues"] = issues

        # 4) symmetry (parallelism)
        ok, issues = self.symmetry.check_document(text)
        report["symmetry"]["ok"] = ok
        report["symmetry"]["issues"] = issues

        # 5) CFG conformance
        ok, issues = self.cfg.check(text)
        report["cfg_research"]["ok"] = ok
        report["cfg_research"]["issues"] = issues

        return report

# ----------------------- Run Analysis -----------------------
def analyze(text: str) -> Dict[str, Any]:
    validator = ResearchValidator()
    return validator.validate(text)

def status_badge(ok: bool) -> str:
    return "✅ OK" if ok else "❌ Issues"

# ----------------------- Main Trigger -----------------------
if run_btn:
    if pdf_file is None and not text_input.strip():
        st.warning("Please upload a PDF or paste the report text.")
        st.stop()

    if pdf_file is not None:
        with st.spinner("Extracting text from PDF…"):
            try:
                raw = extract_text_from_pdf(pdf_file)
            except Exception as e:
                st.error(f"Failed to read PDF: {e}")
                st.stop()
    else:
        raw = text_input

    if not raw or raw.strip() == "":
        st.error("No text found. Please check your input.")
        st.stop()

    with st.spinner("Running PDA-based analysis…"):
        report = analyze(raw)

    # ------------------- Report UI -------------------
    st.subheader("Report")
    cols = st.columns(5)
    sections = [
        ("Balanced Delimiters & Quotes", "balanced_delimiters"),
        ("Section Order & Coverage", "section_order"),
        ("Numbering Hierarchy", "numbering"),
        ("Symmetry / Parallelism", "symmetry"),
        ("Research CFG Conformance", "cfg_research"),
    ]
    for i, (title, key) in enumerate(sections):
        cols[i % 5].metric(title, status_badge(report[key]["ok"]))

    st.markdown("---")

    # Detailed sections
    for title, key in sections:
        with st.expander(f"{title} — {status_badge(report[key]['ok'])}", expanded=not report[key]["ok"]):
            if key == "section_order":
                found = report[key].get("found", [])
                if found:
                    mapping = {'a':'Abstract','i':'Introduction/Background','m':'Methods','r':'Results','d':'Discussion','c':'Conclusion','f':'References'}
                    pretty = [mapping.get(s, s) for s in found]
                    st.markdown("**Detected headings (in order):** " + " → ".join(pretty))
            issues = report[key]["issues"]
            if issues:
                for j, msg in enumerate(issues, 1):
                    st.write(f"{j}. {msg}")
            else:
                st.write("No issues found.")

    # Download JSON
    st.download_button(
        label="💾 Download JSON report",
        file_name="research_validator_report.json",
        mime="application/json",
        data=json.dumps(report, indent=2),
        use_container_width=True
    )

else:
    st.info("Upload a PDF or paste text, then click **Analyze**.")
