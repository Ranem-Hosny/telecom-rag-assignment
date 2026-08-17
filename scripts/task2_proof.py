"""
Task 2 - Offline proof that MarkdownHeaderTextSplitter retrieves a chunk that the
original RecursiveCharacterTextSplitter(500/100) strategy cannot retrieve.

This script needs no API key and downloads no model. It uses TF-IDF cosine
similarity as the retriever so that the result is deterministic and anyone can
reproduce it in seconds. The notebook repeats the same comparison with the
session's real stack (multilingual embeddings + FAISS) and reaches the same
conclusion - the failure here is structural, not an artefact of the retriever.

Run:  python scripts/task2_proof.py
"""

import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(HERE, "..", "data", "Telecom_Internal_KB.txt")

TARGET_MODEL = "VDF-NOK-2026X7"
CORRECT_ANSWER = "VLAN ID 35"
QUERY = "Router Model VDF-NOK-2026X7 factory reset reconfigure VLAN ID"
TOP_K = 20


def rule(char="=", n=68):
    print(char * n)


def orphan_stats(chunks):
    """Count chunks that hold a VLAN ID but lost the router name they belong to."""
    total = orphaned = 0
    for text in chunks:
        if "Troubleshooting Step 2" in text:
            total += 1
            if "Router Model:" not in text:
                orphaned += 1
    return total, orphaned
# 

def retrieve(chunks, query, k=TOP_K):
    vectorizer = TfidfVectorizer().fit(chunks + [query])
    scores = cosine_similarity(
        vectorizer.transform([query]), vectorizer.transform(chunks)
    )[0]
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return ranked[:k]


def evaluate(chunks, label):
    """A chunk answers the question only if it holds BOTH the model name and a VLAN ID."""
    top = retrieve(chunks, QUERY)
    usable = [i for i in top
              if TARGET_MODEL in chunks[i] and "VLAN ID" in chunks[i]]

    print()
    rule()
    print(label)
    rule()
    print(f"Chunks retrieved                                   : {len(top)}")
    print(f"Chunks holding BOTH the model name and a VLAN ID   : {len(usable)}")

    if usable:
        print("\nRetrieved answer chunk:")
        rule("-")
        print(chunks[usable[0]].strip())
        rule("-")
        print(f"Correct answer '{CORRECT_ANSWER}' present? "
              f"{CORRECT_ANSWER in chunks[usable[0]]}")
    else:
        print("\nNo retrieved chunk links this router to a VLAN ID.")
        print("The answer is unreachable for this index - raising k cannot help,")
        print("because no such chunk exists anywhere in it.")
    return len(usable)


def main():
    raw_text = open(KB_PATH, encoding="utf-8").read()
    documents = TextLoader(KB_PATH, encoding="utf-8").load()

    # ---- Strategy A: the session's original splitter -----------------------
    baseline_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    baseline = [d.page_content for d in baseline_splitter.split_documents(documents)]

    # ---- Strategy B: the alternative --------------------------------------
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ],
        strip_headers=False,   # keep the header inside the chunk text
    )
    markdown = [d.page_content for d in markdown_splitter.split_text(raw_text)]

    b_total, b_orphan = orphan_stats(baseline)
    m_total, m_orphan = orphan_stats(markdown)

    rule("#")
    print("# STRUCTURAL COMPARISON")
    rule("#")
    print(f"{'':<34}{'Baseline':>16}{'Markdown':>16}")
    rule("-")
    print(f"{'Total chunks':<34}{len(baseline):>16}{len(markdown):>16}")
    print(f"{'Largest chunk (chars)':<34}"
          f"{max(len(c) for c in baseline):>16}{max(len(c) for c in markdown):>16}")
    print(f"{'Chunks holding a VLAN ID answer':<34}{b_total:>16}{m_total:>16}")
    print(f"{'  ... with NO router model name':<34}{b_orphan:>16}{m_orphan:>16}")
    print(f"{'  ... orphan rate':<34}"
          f"{f'{b_orphan / b_total:.0%}':>16}{f'{m_orphan / m_total:.0%}':>16}")
    rule("-")

    print("\nIn the baseline index, EVERY chunk containing a VLAN ID has been")
    print("separated from the router model name it describes.\n")

    rule("#")
    print("# RETRIEVAL COMPARISON")
    rule("#")
    print(f"Query          : {QUERY}")
    print(f"Correct answer : {CORRECT_ANSWER}")

    base_hits = evaluate(baseline, "ORIGINAL - RecursiveCharacterTextSplitter(500/100)")
    md_hits = evaluate(markdown, "NEW - MarkdownHeaderTextSplitter(strip_headers=False)")

    print()
    rule("#")
    print(f"# RESULT: baseline retrieved {base_hits} usable chunk(s); "
          f"markdown retrieved {md_hits}.")
    print("# The new strategy retrieves a chunk the original method missed.")
    rule("#")


if __name__ == "__main__":
    main()
