"""Get a feel for autoregressive "shortcutting" on Qwen3-4B.

For each question we decode up to N tokens greedily and answer one, clean
diagnostic question: **is the very first token emitted the right answer?**

That is the "autoregression is an accidental interface" check in miniature.
If the answer were already fully resolved in the latent state, the first token
should usually BE the answer. If instead the model "talks its way" to it —
dumps a wrong/neutral first token and only lands on the answer after several
tokens of forward-decode — that is shortcutting through language rather than a
single clean readout of the answer.

Output per row:
    [1st=WRONG sex 200t]  the first *token* (decoded) + whether it is a
                          correct-answer token, up to 200 tokens total.
    [1st=RIGHT yes 7t  ]  first token was already a correct-answer token.

Conventions match bench.py: binary BoolQ prompt "Question: ...\nAnswer:",
greedy (argmax), answer set (Yes, No) with True -> Yes.

Usage:
    python ar_feel.py [N]              # default 8 questions
    python ar_feel.py 8 --max 200
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

MODEL = "Qwen/Qwen3-4B"
ANSWER_SET = ("Yes", "No")
RIGHTS_WINDOW = 8   # how deep into the decode we look for the first
                    # right-answer token; if it isn't there immediately, the
                    # model is "talking its way" to the answer rather than
                    # emitting it as one clean readout.


def fmt_example(r, demos=()):
    """Prompt stem ending in ``Answer:`` (same as bench.py).

    With `demos`, prepend K correct Q/A demonstrations to few-shot the readout
    (Answer: <Yes|No>), so the target inherits the same conditioning.
    """
    passage = (r.get("passage") or "").strip()
    head = ""
    for d in demos:
        dpass = (d.get("passage") or "").strip()
        head += (f"Passage: {dpass}\n" if dpass else "") + \
                f"Question: {d['question']}\nAnswer: " + \
                ("Yes" if bool(d["answer"]) else "No") + "\n\n"
    stem = f"Passage: {passage}\n" if passage else ""
    stem += f"Question: {r['question']}\nAnswer:"
    return head + stem


def answer_tokens(tok):
    """Per-class disjoint token ids for the continuation after 'Answer:'.

    Includes space-prefixed and bare forms, and drops tokens that appear in
    more than one class so the classes stay disjoint (as in bench.py). Returns
    (no_toks, yes_toks) matching ANSWER_SET ordering.
    """
    base = len(tok("Answer:")["input_ids"])
    sets = []
    for ans in ANSWER_SET:
        s = set()
        for variant in (ans, " " + ans):
            s.update(tok("Answer:" + variant)["input_ids"][base:])
        sets.append(s)
    counts = {t: sum(t in s for s in sets) for s in sets for t in s}
    overlap = {t for t, c in counts.items() if c > 1}
    return [s - overlap for s in sets]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("n", type=int, nargs="?", default=8,
                    help="number of questions to decode (default 8)")
    ap.add_argument("--max", type=int, default=200,
                    help="max new tokens to decode per question")
    ap.add_argument("--fewshot", type=int, default=0, metavar="K",
                    help="prepend K correct Q/A demonstrations (default 0=zero shot)")
    ap.add_argument("--tokens", action="store_true",
                    help="show the first emitted tokens one per line")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).to(device).eval()
    print(f"loaded ({args.fewshot}-shot).\n", flush=True)

    no_toks, yes_toks = answer_tokens(tok)
    right_set = yes_toks | no_toks

    from datasets import load_dataset
    ds = load_dataset("google/boolq")
    rows = [dict(r) for r in ds["validation"]][: args.n]
    # Demonstration source: clean train examples, distinct from the eval rows.
    demos = []
    if args.fewshot:
        for d in ds["train"]:
            demos.append(dict(d))  # BoolQ rows always have passage+question
            if len(demos) >= args.fewshot:
                break

    first_right = 0
    never_right = 0           # right-answer token never emitted in decode
    for i, r in enumerate(rows):
        prompt = fmt_example(r, demos)
        gold = "Yes" if bool(r["answer"]) else "No"

        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = ids.shape[1]
        gen = model.generate(
            input_ids=ids,
            max_new_tokens=args.max,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        new_ids = gen[0, prompt_len:].tolist()
        first_id = new_ids[0]
        first_dec = tok.decode([first_id], skip_special_tokens=True).strip()

        gold_toks = yes_toks if gold == "Yes" else no_toks
        if first_id in gold_toks:
            verdict = "RIGHT"
            first_right += 1
        elif first_id in (right_set - gold_toks):
            verdict = "WRONG"
        else:
            verdict = "NEUTRAL"

        # first position in the emitted decode (within the short window) that
        # is a right-answer token; if it's outside the window we treat the
        # model as "talking its way" to the answer rather than reading it out.
        right_at = next((j for j, t in enumerate(new_ids[:RIGHTS_WINDOW])
                         if t in gold_toks), None)
        if not any(t in gold_toks for t in new_ids):
            never_right += 1

        ntok = len(new_ids)
        full = tok.decode(new_ids, skip_special_tokens=True)

        tail = f"   right@pos{right_at}" if right_at is not None else "   right>NOT-IN-WINDOW"
        tail += "   <-- hit cap" if ntok >= args.max else "   <-- self-stopped"
        print(f"\n[{i+1:>2}/{len(rows)}] gold={gold}  1st={verdict:<7} "
              f"'{first_dec}'   ntok={ntok}{tail}")
        print(f"    prompt> {prompt.rstrip()}")
        print(f"    out>    {full!r}")

        if args.tokens:
            pad = lambda s: s.replace(" ", "␣").replace("\n", "⏎").replace("\t", "⇥")
            print(f"    tokens> (first {min(RIGHTS_WINDOW, ntok)} of {ntok})")
            for k, t in enumerate(new_ids[:RIGHTS_WINDOW]):
                dec = pad(tok.decode([t], skip_special_tokens=True))
                mark = ""
                if t in gold_toks and k == right_at:
                    mark = "  <== FIRST RIGHT-TOK" if k == 0 else "  <== RIGHT-TOK"
                elif k == 0:
                    mark = "  <== FIRST"
                print(f"        {k:<3} {t:<8} {dec!r}{mark}")

    print(f"\n=== first-token-is-answer: {first_right}/{len(rows)} "
          f"right-tok-ever: {len(rows) - never_right}/{len(rows)} "
          f"never-right: {never_right}/{len(rows)} "
          f"(cap={args.max}) model={MODEL} ===")


if __name__ == "__main__":
    main()