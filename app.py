import os
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import AsyncOpenAI
import anthropic
from google import genai

load_dotenv()

st.set_page_config(
    page_title="AI Debate Arena 1.1",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

AI_NAMES = ["GPT", "Claude", "Gemini", "Kimi", "Grok"]

DEFAULT_MODELS = {
    "GPT": os.getenv("GPT_MODEL", "gpt-5.6"),
    "Claude": os.getenv("CLAUDE_MODEL", "claude-opus-4-8"),
    "Gemini": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    "Kimi": os.getenv("KIMI_MODEL", "kimi-k3"),
    "Grok": os.getenv("GROK_MODEL", "grok-4.7"),
}

SYSTEMS = {
    "GPT": """You are GPT, an equal participant in a five-model deliberation.
You must reason independently first, then debate. Never treat majority opinion as
proof. Distinguish facts, inference, assumptions, and value judgments. You may
change your mind, but only for a stated reason. Never fabricate evidence.""",
    "Claude": """You are Claude, an equal participant in a five-model deliberation.
Critically test premises, evidence, logic and uncertainty. Do not agree socially or
because of majority pressure. You may accept, reject, or remain uncertain about a
claim. Explain every change of position. Never fabricate evidence.""",
    "Gemini": """You are Gemini, an equal participant in a five-model deliberation.
Compare evidence and reasoning rather than vote counts. Explicitly separate what is
established, plausible, disputed, and unknown. You can revise your own claims. Never
invent citations or facts.""",
    "Kimi": """You are Kimi, an equal participant in a five-model deliberation.
Challenge weak assumptions and unsupported claims. Accept another model's point only
when justified. It is valid to remain UNCERTAIN. State what evidence would resolve
the disagreement. Never manufacture evidence.""",
    "Grok": """You are Grok, an equal participant in a five-model deliberation.
Be rigorous and adversarial toward arguments, not people. Look for contradictions,
missing evidence, and overconfidence. Do not confuse consensus with truth. Revise
your position when warranted and explain why. Never fabricate evidence.""",
}

CSS = """
<style>
.block-container {max-width:1500px;padding-top:1.3rem}
.consensus {border:2px solid #16a34a;border-radius:12px;padding:14px;background:#f0fdf4;margin:8px 0}
.unresolved {border:2px solid #f59e0b;border-radius:12px;padding:14px;background:#fffbeb;margin:8px 0}
.info-card {border:1px solid #d1d5db;border-radius:12px;padding:14px;background:#fafafa;margin:8px 0}
.badge {display:inline-block;padding:3px 8px;border-radius:999px;font-size:.78rem;font-weight:700}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------- API ----------------

async def openai_call(model, system, prompt, key, base_url=None):
    client = AsyncOpenAI(api_key=key, base_url=base_url)
    r = await client.responses.create(model=model, instructions=system, input=prompt)
    return r.output_text


async def claude_call(model, system, prompt, key):
    client = anthropic.AsyncAnthropic(api_key=key)
    r = await client.messages.create(
        model=model,
        max_tokens=7000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(x.text for x in r.content if hasattr(x, "text"))


async def gemini_call(model, system, prompt, key):
    client = genai.Client(api_key=key)
    r = await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=prompt,
        config={"system_instruction": system, "max_output_tokens": 7000},
    )
    return r.text


async def call_ai(name, model, prompt):
    if name == "GPT":
        return await openai_call(model, SYSTEMS[name], prompt, os.environ["OPENAI_API_KEY"])
    if name == "Claude":
        return await claude_call(model, SYSTEMS[name], prompt, os.environ["ANTHROPIC_API_KEY"])
    if name == "Gemini":
        return await gemini_call(model, SYSTEMS[name], prompt, os.environ["GEMINI_API_KEY"])
    if name == "Kimi":
        return await openai_call(
            model, SYSTEMS[name], prompt, os.environ["KIMI_API_KEY"],
            os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
        )
    if name == "Grok":
        return await openai_call(
            model, SYSTEMS[name], prompt, os.environ["XAI_API_KEY"],
            "https://api.x.ai/v1"
        )
    raise ValueError(name)


async def parallel(configs, prompt):
    async def one(name):
        t = time.perf_counter()
        try:
            out = await call_ai(name, configs[name], prompt)
            return name, {"ok": True, "text": out, "seconds": time.perf_counter() - t}
        except Exception as e:
            return name, {"ok": False, "text": "", "error": str(e),
                          "seconds": time.perf_counter() - t}

    return dict(await asyncio.gather(*(one(n) for n in AI_NAMES)))


def run_parallel(configs, prompt):
    return asyncio.run(parallel(configs, prompt))


# ---------------- Prompt helpers ----------------

def pack(results):
    return "\n\n".join(
        f"===== {name} =====\n{results[name].get('text', '[NO RESPONSE]')}"
        for name in AI_NAMES
    )


def independent_prompt(q, style):
    return f"""
QUESTION:
{q}

MODE:
{style}

Give your independent answer BEFORE seeing any other AI.
Include:
- conclusion
- strongest reasons
- important assumptions
- uncertainty
- what evidence could change your mind

Do not invent facts or sources.
"""


def debate_prompt(q, initial, history, round_no, style):
    return f"""
QUESTION:
{q}

MODE:
{style}

INDEPENDENT ANSWERS:
{initial}

PREVIOUS DEBATE:
{history if history else "[none]"}

DEBATE ROUND:
{round_no}

Debate as an equal participant.

You must:
1. Directly address important claims from the other models.
2. Identify errors, weak evidence, assumptions, contradictions, and missing context.
3. Defend claims that remain strong.
4. Admit and correct mistakes.
5. Do not agree because of majority pressure.
6. Do not manufacture evidence.
7. End with:
   - strongest points you still defend
   - points you now reject/revise
   - unresolved questions

This is still a debate, NOT a judge stage.
"""


def assessment_prompt(q, initial, debate, style):
    return f"""
QUESTION:
{q}

MODE:
{style}

INITIAL ANSWERS:
{initial}

LATEST DEBATE:
{debate}

Now perform a structured post-debate assessment.

Return exactly these sections:

KELEBIHAN:
- What each model contributed that is useful/strong.
- Include the model name.

KEKURANGAN:
- What is weak, unsupported, contradictory, or incomplete.
- Include the model name.

KESAMAAN:
- Substantive claims or reasoning that multiple models independently share.

PERBEDAAN:
- Important claims where models disagree.

KOREKSI:
- Specific claims that should be corrected, withdrawn, or qualified.
- Explain why.

DO NOT decide by majority. A claim being repeated by five models does not make
it true. Focus on evidence and reasoning.
"""


def candidate_prompt(q, initial, debate, assessment):
    return f"""
QUESTION:
{q}

INITIAL ANSWERS:
{initial}

DEBATE:
{debate}

POST-DEBATE ASSESSMENT:
{assessment}

Create a list of concrete CLAIMS that can be checked by all participants.

Each claim must be:
- specific
- independently understandable
- neither a vague summary nor a compound claim
- phrased so an AI can ACCEPT, REJECT, or UNCERTAIN

Do not decide which claims are correct. This is only a neutral candidate list.

Use:
C1: ...
C2: ...
C3: ...
"""


def consensus_review_prompt(q, candidates, assessment, prior):
    return f"""
QUESTION:
{q}

CANDIDATE CLAIMS:
{candidates}

POST-DEBATE ASSESSMENT:
{assessment}

PREVIOUS CONSENSUS STATE:
{prior if prior else "[first consensus check]"}

You are independently reviewing each candidate claim.

For EVERY claim, output exactly one status:
ACCEPT
REJECT
UNCERTAIN

Rules:
- ACCEPT only if you personally find the claim sufficiently supported.
- REJECT if you believe it is false, materially misleading, or unsupported.
- UNCERTAIN if evidence is insufficient or the wording cannot safely be accepted.
- You are NOT required to agree with other AIs.
- Never accept a claim simply because most models accept it.
- Explain your reason briefly.
- If a claim needs narrower wording, say exactly how it should be rewritten.

Format:
C1 | STATUS | reason | suggested wording if needed
C2 | STATUS | reason | suggested wording if needed
"""


def revision_prompt(q, candidates, reviews, assessment, round_no):
    return f"""
QUESTION:
{q}

CANDIDATE CLAIMS:
{candidates}

INDIVIDUAL CONSENSUS REVIEWS:
{reviews}

ASSESSMENT:
{assessment}

This is CONSENSUS REVISION ROUND {round_no}.

Re-examine the disputed claims. Address the specific objections made by the other
models. You may:
- defend the claim,
- narrow/rewrite it,
- reject it,
- or identify evidence that is still missing.

Do NOT force agreement. The goal is to determine which exact propositions can
honestly be accepted by everyone.

End with your revised position on each disputed claim.
"""


def final_individual_prompt(q, accepted, unresolved, all_debate):
    return f"""
QUESTION:
{q}

CLAIMS ACCEPTED BY EVERY PARTICIPANT:
{accepted}

CLAIMS STILL UNRESOLVED:
{unresolved}

DEBATE RECORD:
{all_debate}

Write your final contribution.

For accepted claims, explain why you now accept them.
For unresolved claims, state your own position and why consensus has not been reached.
Do not pretend there is consensus where there is none.
"""


# ---------------- Session helpers ----------------

def save_json(data):
    return json.dumps(data, ensure_ascii=False, indent=2)


def api_status():
    return {
        "GPT": bool(os.getenv("OPENAI_API_KEY")),
        "Claude": bool(os.getenv("ANTHROPIC_API_KEY")),
        "Gemini": bool(os.getenv("GEMINI_API_KEY")),
        "Kimi": bool(os.getenv("KIMI_API_KEY")),
        "Grok": bool(os.getenv("XAI_API_KEY")),
    }


# ---------------- UI ----------------

st.title("⚔️ AI Debate Arena 1.1")
st.caption("Independent answers → debate → strengths/weaknesses/common ground → correction → unanimous consensus loop")

with st.sidebar:
    st.header("Control Room")
    max_debate = st.slider("Ronde debat awal", 1, 8, 3)
    max_consensus = st.slider("Maks. ronde konsensus", 1, 10, 5)
    style = st.selectbox(
        "Gaya debat",
        ["Evidence-first", "Adversarial", "Balanced", "Academic"],
    )

    st.divider()
    st.subheader("Model IDs")
    configs = {}
    for name in AI_NAMES:
        configs[name] = st.text_input(
            name, os.getenv(f"{name.upper()}_MODEL", {
                "GPT":"gpt-5.6", "Claude":"claude-opus-4-8",
                "Gemini":"gemini-3.8-flash", "Kimi":"kimi-k3", "Grok":"grok-4.7"
            }[name])
        )

    st.divider()
    statuses = api_status()
    for n in AI_NAMES:
        if statuses[n]:
            st.success(f"{n}: API OK")
        else:
            st.error(f"{n}: API belum tersedia")

    st.divider()
    upload = st.file_uploader("Muat sesi .json", type=["json"])
    if upload and st.button("Muat sesi"):
        st.session_state["loaded"] = json.loads(upload.getvalue().decode())
        st.rerun()

question = st.text_area(
    "Pertanyaan",
    value=st.session_state.get("question", ""),
    height=120,
    placeholder="Masukkan pertanyaan yang ingin diuji sampai mencapai kesepakatan yang benar-benar dapat dipertanggungjawabkan."
)

start = st.button("🚀 MULAI DEBAT & CONSENSUS", type="primary", use_container_width=True)

if start:
    if not question.strip():
        st.error("Pertanyaan belum diisi.")
        st.stop()

    missing = [n for n in AI_NAMES if not statuses[n]]
    if missing:
        st.error("API key belum tersedia: " + ", ".join(missing))
        st.stop()

    st.session_state.clear()
    st.session_state["question"] = question
    st.session_state["configs"] = configs

    # 1. Independent
    st.header("01 · Jawaban Independen")
    initial = run_parallel(configs, independent_prompt(question, style))
    st.session_state["initial"] = initial

    tabs = st.tabs(AI_NAMES)
    for tab, n in zip(tabs, AI_NAMES):
        with tab:
            r = initial[n]
            if r["ok"]:
                st.markdown(r["text"])
                st.caption(f"{configs[n]} · {r['seconds']:.1f}s")
            else:
                st.error(r["error"])

    initial_text = pack(initial)

    # 2. Initial debate
    debate_results = []
    history = ""
    st.header("02 · Debat")
    for rr in range(1, max_debate + 1):
        with st.spinner(f"Debat ronde {rr}/{max_debate}..."):
            res = run_parallel(
                configs,
                debate_prompt(question, initial_text, history, rr, style)
            )
        debate_results.append(res)
        history += f"\n\n===== ROUND {rr} =====\n{pack(res)}"

        st.subheader(f"Ronde {rr}")
        tabs = st.tabs(AI_NAMES)
        for tab, n in zip(tabs, AI_NAMES):
            with tab:
                if res[n]["ok"]:
                    st.markdown(res[n]["text"])
                else:
                    st.error(res[n]["error"])

    # 3. Strengths / weaknesses / common ground / corrections
    st.header("03 · Kelebihan, Kekurangan, Kesamaan & Koreksi")
    latest_debate = pack(debate_results[-1])
    with st.spinner("Kelima AI melakukan post-debate assessment..."):
        assessments = run_parallel(
            configs, assessment_prompt(question, initial_text, latest_debate, style)
        )

    st.session_state["assessments"] = assessments

    tabs = st.tabs(AI_NAMES)
    for tab, n in zip(tabs, AI_NAMES):
        with tab:
            if assessments[n]["ok"]:
                st.markdown(assessments[n]["text"])
            else:
                st.error(assessments[n]["error"])

    # 4. Candidate claims
    with st.spinner("Menyusun kandidat klaim untuk diuji bersama..."):
        # Each AI proposes candidates independently; all proposals are pooled.
        candidate_results = run_parallel(
            configs, candidate_prompt(question, initial_text, latest_debate, pack(assessments))
        )
    candidate_text = pack(candidate_results)

    # 5. Consensus loop
    st.header("04 · Consensus Loop")
    st.info(
        "Aturan: klaim masuk konsensus hanya jika kelima AI secara eksplisit ACCEPT. "
        "REJECT atau UNCERTAIN dari satu AI membuat klaim kembali ke pembaruan."
    )

    consensus_history = []
    accepted = []
    unresolved = []
    current_candidates = candidate_text
    prior_state = ""

    for cr in range(1, max_consensus + 1):
        st.subheader(f"Consensus Round {cr}/{max_consensus}")

        reviews = run_parallel(
            configs,
            consensus_review_prompt(
                question, current_candidates, pack(assessments), prior_state
            )
        )
        consensus_history.append({
            "round": cr,
            "reviews": reviews,
            "candidates": current_candidates,
        })

        # A deterministic, transparent gate:
        # a claim is unanimously accepted only if EVERY review explicitly contains
        # ACCEPT for that exact candidate label. We use the candidate IDs as the gate.
        candidate_ids = []
        for line in current_candidates.splitlines():
            s = line.strip()
            if s.startswith("C") and ":" in s:
                candidate_ids.append(s.split(":", 1)[0].strip())

        status_matrix = {cid: [] for cid in candidate_ids}
        for n in AI_NAMES:
            txt = reviews[n].get("text", "") if reviews[n]["ok"] else ""
            for cid in candidate_ids:
                found = None
                for line in txt.splitlines():
                    if line.strip().startswith(cid + " |"):
                        upper = line.upper()
                        if "| ACCEPT" in upper:
                            found = "ACCEPT"
                        elif "| REJECT" in upper:
                            found = "REJECT"
                        elif "| UNCERTAIN" in upper:
                            found = "UNCERTAIN"
                        break
                status_matrix[cid].append(found or "UNCERTAIN")

        accepted = [cid for cid, vals in status_matrix.items()
                    if len(vals) == 5 and all(v == "ACCEPT" for v in vals)]
        unresolved = [cid for cid, vals in status_matrix.items()
                      if cid not in accepted]

        st.write("**Status klaim:**")
        for cid, vals in status_matrix.items():
            if cid in accepted:
                st.markdown(
                    f'<div class="consensus"><b>{cid}</b> — ✓ ACCEPTED BY ALL<br>'
                    f'GPT={vals[0]} · Claude={vals[1]} · Gemini={vals[2]} · '
                    f'Kimi={vals[3]} · Grok={vals[4]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="unresolved"><b>{cid}</b> — masih diperdebatkan<br>'
                    f'GPT={vals[0]} · Claude={vals[1]} · Gemini={vals[2]} · '
                    f'Kimi={vals[3]} · Grok={vals[4]}</div>',
                    unsafe_allow_html=True
                )

        if not unresolved:
            st.success("Semua klaim kandidat yang diuji telah diterima oleh kelima AI.")
            break

        # Revision: all five see all reviews and disputed claims, then independently
        # revise the disputed points.
        disputed_text = "\n".join(
            f"{cid}: statuses={status_matrix[cid]}" for cid in unresolved
        )

        with st.spinner(f"Memperbarui argumen untuk {len(unresolved)} klaim yang belum konsensus..."):
            revisions = run_parallel(
                configs,
                revision_prompt(
                    question, current_candidates, pack(reviews),
                    pack(assessments), cr
                )
            )

        consensus_history.append({
            "revision_round": cr,
            "revisions": revisions,
            "disputed": disputed_text,
        })

        # New candidate set: pool revised formulations from all five, with explicit
        # instruction to focus on unresolved issues.
        current_candidates = "\n\n".join(
            f"===== {n} REVISION =====\n{revisions[n].get('text', '')}"
            for n in AI_NAMES
        )
        prior_state = "\n".join(
            f"{cid}: {status_matrix[cid]}" for cid in unresolved
        )

    # 6. Final independent position, not a judge
    st.header("05 · Posisi Akhir Masing-Masing AI")

    accepted_text = "\n".join(accepted) if accepted else "[Belum ada klaim yang lolos unanimous gate]"
    unresolved_text = "\n".join(unresolved) if unresolved else "[Tidak ada klaim tersisa pada gate terakhir]"

    final_prompt_text = final_individual_prompt(
        question, accepted_text, unresolved_text, history
    )
    finals = run_parallel(configs, final_prompt_text)
    st.session_state["finals"] = finals

    tabs = st.tabs(AI_NAMES)
    for tab, n in zip(tabs, AI_NAMES):
        with tab:
            if finals[n]["ok"]:
                st.markdown(finals[n]["text"])
            else:
                st.error(finals[n]["error"])

    # Save
    payload = {
        "version": "1.1",
        "created_at": datetime.now().isoformat(),
        "question": question,
        "models": configs,
        "debate_rounds": max_debate,
        "max_consensus_rounds": max_consensus,
        "style": style,
        "initial": initial,
        "debates": debate_results,
        "assessments": assessments,
        "candidate_generation": candidate_results,
        "consensus_history": consensus_history,
        "finals": finals,
        "unanimously_accepted_claim_ids": accepted,
        "unresolved_claim_ids_at_last_gate": unresolved,
    }
    st.session_state["session_json"] = json.dumps(payload, ensure_ascii=False, indent=2)

    st.divider()
    st.download_button(
        "💾 Download seluruh sesi",
        data=st.session_state["session_json"],
        file_name="ai_debate_arena_1.1_session.json",
        mime="application/json",
        use_container_width=True,
    )

# Load session display
if "loaded" in st.session_state:
    data = st.session_state.pop("loaded")
    st.session_state["question"] = data.get("question", "")
    st.success(
        f"Sesi v{data.get('version','?')} dimuat. "
        f"Gunakan file JSON sebagai arsip/transkrip; jalankan ulang jika ingin meneruskan."
    )
