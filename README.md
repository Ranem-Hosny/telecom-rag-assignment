# Telecom RAG — Day 2 Assignment

Solution for the two assignment tasks, built on top of the session notebook
`01_telecom_rag_demo.ipynb`.

- **Task 1** — Prompt Engineering Challenge: 3 system prompts using negative constraints
- **Task 2** — Chunking Strategy Challenge: `MarkdownHeaderTextSplitter` vs the original splitter

Everything reuses the session's stack — the same knowledge base
(`Telecom_Internal_KB.txt`), the same embedding model
(`paraphrase-multilingual-MiniLM-L12-v2`), the same vector store (FAISS) and the
same LLM (`gemini-2.5-flash`, `temperature=0`). Holding all of that constant
means every measured difference comes from the prompt or the chunking strategy
and from nothing else.

---

## Repository layout

```
.
├── README.md                            this file — the write-up for both tasks
├── requirements.txt
├── data/
│   ├── Telecom_Internal_KB.txt          knowledge base (from the session)
│   └── Telecom_Internal_KB.pdf
├── notebooks/
│   ├── 01_telecom_rag_demo.ipynb        the original session notebook
│   ├── 02_assignment_solution.ipynb     the solution (run locally)
│   └── RAG_Assignment_COLAB.ipynb       same solution, self-contained for Colab
├── scripts/
│   └── task2_proof.py                   offline proof — no API key, no downloads
└── results/
    └── task2_proof_output.txt           actual output of the script above
```

## Running it

**Fastest — the offline proof for Task 2** (no API key, runs in seconds):

```bash
pip install -r requirements.txt
python scripts/task2_proof.py
```

**Full run, including Task 1** — open `notebooks/02_assignment_solution.ipynb`
and add your Gemini key to `notebooks/.env`:

```
GOOGLE_API_KEY = "your-key-here"
```

**On Colab** — upload `notebooks/RAG_Assignment_COLAB.ipynb`, run all, and give it
the KB file and the key when prompted.

---

# Task 1 — Prompt Engineering with Negative Constraints

## Approach

A positive constraint says what the model should do. A negative constraint says
what it must **never** do.

This distinction matters here because the expensive failures in a support
assistant are not missing answers — they are **confident wrong answers**:
inventing a price, granting a compensation the policy forbids, or confirming the
company's identity. Those are behaviours a positive instruction does not prevent.

Each prompt was tested against three tickets, chosen to trigger a specific
failure rather than the happy path:

| # | Ticket | Trap |
|---|--------|------|
| A | Asks the monthly price of the 200 Mbps package | Pricing does not exist in the KB → **hallucination** |
| B | Demands compensation after a 36-hour outage | Policy allows it only above 72 hours → **policy violation** |
| C | Asks the agent to confirm the company is Vodafone | Naming real brands is forbidden → **instruction leakage** |

## V1 — the session's original prompt

**Negative constraints: 1**

> `إياك أن تذكر أي اسم شركة اتصالات حقيقي`

Everything else is phrased positively — *"use the internal context"*, *"tell the
customer if a technician is needed"*. Instructing the model to use the context is
a **preference**, not a prohibition, so when the context has no answer nothing
stops it from falling back on its pre-training.

**Behaviour:** holds on ticket C, the one thing it was explicitly told not to do.
Fails on A — it produces plausible-looking pricing that appears nowhere in the
knowledge base. Unreliable on B, because it was asked to be a helpful agent and
granting the compensation is the helpful-sounding move.

## V2 — grounding constraints added

**Negative constraints: 6**

The additions that matter:

- `ممنوع تستخدم أي معلومة من معرفتك العامة` — closes the pre-training fallback
- `ممنوع تذكر أي رقم غير مكتوب حرفيًا في السياق` — closes invented figures specifically
- `ممنوع توعد العميل بأي تعويض إلا لو الشروط متحققة` — closes the policy violation
- `ممنوع تخمّن أو تفترض` — closes gap-filling
- `لو المعلومة مش موجودة، ممنوع تحاول تجاوب` — gives the model a licensed exit

**Change:** A and B are fixed. Notably, constraint #3 does more work than #2 —
banning *general knowledge* is abstract, while banning *any number not written in
the context* is concrete and checkable, and compliance with the concrete version
is far more reliable.

**Residual problem:** V2 is factually grounded but not presentable. It leaks
internal vocabulary to the customer (`حسب السياق الداخلي`) and its refusals are
long and inconsistently worded, which makes them impossible to QA at scale.

## V3 — behavioural and formatting constraints added

**Negative constraints: 9**

V2's six, plus:

- `ممنوع تقول "السياق" أو "المستندات"` — stops internal wording reaching the customer
- `ممنوع تكرر الاعتذار` / `ممنوع مقدمة طويلة` — stops rambling
- `ممنوع الرد يزيد عن 5 أسطر` — enforces a support-appropriate length
- a single **fixed refusal sentence**, quoted verbatim in the prompt

**Change:** same factual accuracy as V2, but the output is customer-ready. The
fixed refusal sentence is the single most valuable line: it turns *"the model
declined somehow"* into a deterministic string that QA can grep for and count.

## Result

| | V1 | V2 | V3 |
|---|---|---|---|
| Negative constraints | 1 | 6 | 9 |
| A — invented pricing | fails | fixed | fixed |
| B — compensation outside policy | unreliable | fixed | fixed |
| C — named a real brand | holds | holds | holds |
| Customer-ready output | no | no | yes |

**Conclusion: a model will not infer a prohibition that was never written.**
V1's single constraint protected exactly the one thing it named and nothing else.
Each failure disappeared only once a constraint named it explicitly — and
concrete, checkable prohibitions outperform abstract ones.

Full prompt/response pairs are written to `task1_results.md` when the notebook runs.

---

# Task 2 — Alternative Chunking Strategy

**Strategy implemented:** `MarkdownHeaderTextSplitter` with `strip_headers=False`,
replacing `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)`.

## The defect in the original strategy

The knowledge base stores each router as a 7-line block of roughly 700 characters:

```
### Router Model: VDF-NOK-2026X7
- **Manufacturer:** Nokia
- **Max Supported Speed:** 200 Mbps
- **DSL Light Behavior:** ...
- **Internet Light Behavior:** ...
- **Troubleshooting Step 1:** Restart router and wait 2 minutes.
- **Troubleshooting Step 2:** Factory reset ... Reconfigure with VLAN ID 35.
```

At `chunk_size=500` every block is cut in two, and the cut consistently lands
between Step 1 and Step 2 — separating the **VLAN ID** from the **model name** it
belongs to. `chunk_overlap=100` cannot bridge the gap, because the model name
sits roughly 600 characters upstream of the VLAN ID.

## Measured result

Reproduce with `python scripts/task2_proof.py`; captured in
`results/task2_proof_output.txt`.

| | Original (Recursive 500/100) | New (MarkdownHeaderTextSplitter) |
|---|---|---|
| Total chunks | 481 | 502 |
| Largest chunk | 495 chars | 740 chars |
| Chunks holding a VLAN ID answer | 200 | 200 |
| ...with **no** router model name | **200 (100%)** | **0 (0%)** |

The failure rate is 100%, not an unlucky edge case. Under the original strategy
**no chunk in the entire index links any router model to its VLAN ID.**

## Retrieval proof

Test query — a realistic agent question:

> *Customer has a VDF-NOK-2026X7 router and did a factory reset. Which VLAN ID
> should it be reconfigured with?*

Correct answer from the source document: **VLAN ID 35**.

A chunk answers this only if it contains **both** the model name and a VLAN ID.
Counting those in each retriever's top 20:

| | Usable chunks retrieved |
|---|---|
| Original strategy | **0** |
| New strategy | **1** — containing `VDF-NOK-2026X7` and `VLAN ID 35` together |

The original retriever returns the chunk naming `VDF-NOK-2026X7` (which stops at
Step 1) and, separately, orphaned Step 2 fragments belonging to other routers.
Raising `k` cannot fix this — the required chunk does not exist in that index at all.

## Why it works

`strip_headers=False` is the decisive setting. It keeps `### Router Model: ...`
inside the chunk text rather than moving it to metadata, so the model name is part
of what gets embedded and stays reachable by semantic search.

No secondary character-level split is applied. The largest markdown chunk is 740
characters — comfortably inside the embedding model's limit — and adding a
500-character split afterwards would immediately reintroduce the original bug.

## General lesson

A fixed character window has no concept of a *record*. When a document is a list
of records — routers, error codes, SKUs, policies — chunk on the record boundary.
The structure was already present in the source; the original pipeline discarded
it at ingestion and paid for it at retrieval time.

## Honest limitation

This strategy depends on the document being well-formed Markdown. The file is
named `.txt`, but its contents use `#`/`##`/`###`/`####` consistently, which is
what makes header splitting viable. On an unstructured PDF the same splitter would
emit one enormous chunk, and **Semantic Chunking** would be the right alternative
to reach for there.

## Note on the offline proof

`scripts/task2_proof.py` uses TF-IDF cosine similarity instead of the session's
embedding model, so that the result is deterministic and reproducible in seconds
without an API key or a model download. The notebook repeats the same comparison
with the real stack — multilingual embeddings and FAISS — and reaches the same
conclusion. The failure is structural: the answer chunk is absent from the index,
so no retriever of any kind can return it.
#   t e l e c o m - r a g - a s s i g n m e n t  
 