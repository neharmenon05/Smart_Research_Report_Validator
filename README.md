# 📑 Smart Research Report Validator (PDA + CFG)

A **Pushdown Automata (PDA) and Context-Free Grammar (CFG) based analyzer** for validating **academic research reports**.  
This tool checks **structural correctness, section ordering, delimiter balance, numbering hierarchy, and symmetry in phrasing** to ensure reports follow standard research writing conventions.

---

## 🎯 Problem Statement
Research reports must adhere to strict **structural and grammatical rules**.  
Common issues include:
- ❌ Unbalanced delimiters `()[]{}` or quotation marks.
- ❌ Missing or misplaced sections (e.g., Methods after Results).
- ❌ Incorrect numbering hierarchy (e.g., jumping from 1.1 → 1.3).
- ❌ Asymmetry in conjunction-based phrasing (e.g., *clarity and are precise*).
- ❌ CFG violations in standard report sequence.

Such errors reduce readability and may cause reports to be rejected in **peer review**.

---

## 🧠 Objectives
This system validates research documents for:
1. ✅ **Balanced Delimiters & Quotes**  
2. ✅ **Section Order & Coverage**  
3. ✅ **Numbering Hierarchy**  
4. ✅ **Symmetry & Parallelism** in phrasing  
5. ✅ **CFG Conformance** (standard report structure)  

---

## 🏗️ System Architecture

**Input** → Research report (PDF file)  
**Processing Pipeline**:
1. 📥 **PDF Extraction** → Extract text using `pdfplumber`.  
2. 🔍 **Preprocessing** → Tokenize into sections, sentences, and tokens.  
3. 🧩 **Validation Modules**:
   - **Balanced Delimiters & Quotes**
   - **Section Order & Coverage**
   - **Numbering Hierarchy**
   - **Symmetry & Parallelism**
   - **CFG Conformance**
4. 🔎**AI Suggestions and summary** → Detailed insights into writing style,content quality,etc
5. 📊 **Report Generation** → Summarized analysis with ✅/❌ results.  
**Output** → Human-readable analysis via **Streamlit UI**.

---

## 🧩 Modules Overview

### 🔹 Module 1: Balanced Delimiters & Quotes
- Validates parentheses `()`, braces `{}`, brackets `[]`, and quotation marks.  
- Uses a **stack-based PDA**.  
- Flags **mismatched or unclosed delimiters**.

### 🔹 Module 2: Section Order & Coverage
- Detects major report sections.  
- Ensures correct order:  
  `Abstract → Introduction → Methods → Results → Discussion → Conclusion → References`.  
- Flags missing or misplaced sections.

### 🔹 Module 3: Numbering Hierarchy
- Validates hierarchical numbering (1, 1.1, 1.2, 2, 2.1 …).  
- Detects skipped or malformed numbering.

### 🔹 Module 4: Symmetry & Parallelism
- Analyzes phrases around conjunctions (*and/or*).  
- Ensures **parallel structure** (noun ↔ noun, verb ↔ verb).  
- Detects asymmetry in phrasing.

### 🔹 Module 5: CFG Conformance
- Implements a **Context-Free Grammar** for research papers

### 🔹 Module 6: AI Summary and Suggestions
- Implements a AI (using api key) providing detailed analysis on **quality of research papers and gives suggestions** for improvement.

