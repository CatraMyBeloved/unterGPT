#!/usr/bin/env python3
"""
prepare_chat_data.py
--------------------
Turn a raw Twitch chat .jsonl scrape into a training-ready dataset for an
Unsloth/QLoRA fine-tune that imitates the chat.

Design (per project spec):
  * CONTEXT  = last N messages (default 32), formatted "user: msg".
               Keeps EVERYTHING, including bot commands (!foo) and bot-account
               messages (Nightbot) -- this is the authentic scroll the live bot
               will see.
  * TARGET   = the single next chat message, BARE (no "user:" prefix).
               Only "real" human messages are target-eligible:
                 - not a bot command (does not start with '!')
                 - not from a known bot account (nightbot, etc.)
  * SESSIONS = the log is split wherever there is a time gap larger than
               --session-gap seconds (default 600 = 10 min). Windows never
               cross a session boundary.
  * WINDOWS  = stride-1 sliding window. For every target-eligible message we
               emit one sample using up to N preceding messages (same session)
               as context. A target needs >=1 context message.

Output: a JSONL where each line has:
    {
      "text":   "<full formatted training string incl. target>",
      "prompt": "<everything up to and including '<next>'>",   # for loss masking
      "completion": "<target msg + stop delimiter>"            # the part to train on
    }

The split point between prompt and completion lets you mask loss to the
completion only (recommended). "text" is the concatenation, provided for
trainers that take a single field.

Format of one sample's text:

    <info>
    you are a viewer in this twitch chat. your name is {BOT_NAME}.
    chat is fast, casual, full of emotes and slang. write one short message.
    </info>
    <chat>
    user1: KEKW
    user2: he really did that poroAgony
    user3: !agaslotmachine
    Nightbot: @user3 You rolled: [ aba | aba | aba ]
    </chat>
    <next>that emote KEKW</next>

Delimiters are configurable. Whatever you pick here MUST match your Ollama
Modelfile TEMPLATE and stop tokens at serve time, byte-for-byte.
"""

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path


# Zero-width / invisible chars commonly appended to dodge Twitch's
# "duplicate message" filter. Stripping them collapses the fake copypasta
# variants back into one string AND stops the model learning to emit garbage.
INVISIBLE = "͏​‌‍⁠ㅤᅟᅠ឴឵﻿᠎"
_INVISIBLE_SET = set(INVISIBLE)


def strip_invisible(s: str) -> str:
    """Drop format (Cf) and control (Cc) category chars plus the explicit list,
    then collapse the whitespace the removals may leave behind."""
    s = "".join(
        c for c in s
        if c not in _INVISIBLE_SET
        and unicodedata.category(c) not in ("Cf", "Cc")
    )
    # removals can leave doubled/edge spaces; normalise to single spaces
    return re.sub(r"\s+", " ", s).strip()


# Accounts whose messages are kept in context but never used as a target.
BOT_ACCOUNTS = {
    "nightbot", "streamelements", "moobot", "fossabot",
    "wizebot", "streamlabs", "sery_bot", "pretzelrocks",
}


def is_command(msg: str) -> bool:
    """A chat/bot command, e.g. !agaslotmachine. Kept in context, never a target."""
    return msg.strip().startswith("!")


def is_bot_account(user: str) -> bool:
    return user.strip().lower() in BOT_ACCOUNTS


def target_eligible(rec: dict) -> bool:
    """True if this message may be used as a prediction target."""
    msg = rec["msg"].strip()
    if not msg:
        return False
    if is_command(msg):
        return False
    if is_bot_account(rec["user"]):
        return False
    return True


# @mentions, e.g. "@nelsoazhang yeah no". Used to detect "X tagged Y, Y replied".
MENTION_RE = re.compile(r"@([A-Za-z0-9_]+)")


def strip_mentions(text: str) -> str:
    """Drop @username tokens (the addressing 'tag') and re-collapse whitespace."""
    return re.sub(r"\s+", " ", MENTION_RE.sub("", text)).strip()


def load_messages(path: Path) -> list[dict]:
    msgs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # normalise: require ts, user, msg
            if "msg" not in rec or "user" not in rec:
                continue
            # strip invisible junk from BOTH user and msg (cleans context too)
            msg = strip_invisible(str(rec["msg"]))
            if not msg:
                continue  # message was nothing but invisible chars -> drop
            # keep ts if present, else None (those just won't trigger gap splits)
            msgs.append({
                "ts": rec.get("ts"),
                "user": strip_invisible(str(rec["user"])),
                "msg": msg,
            })
    return msgs


def split_sessions(msgs: list[dict], gap_seconds: int) -> list[list[dict]]:
    """Split the (time-ordered) message list into sessions on large time gaps."""
    if not msgs:
        return []
    # sort by ts when available; messages without ts keep original order at the end
    timed = [m for m in msgs if isinstance(m["ts"], (int, float))]
    untimed = [m for m in msgs if not isinstance(m["ts"], (int, float))]
    timed.sort(key=lambda m: m["ts"])

    sessions = []
    cur = []
    prev_ts = None
    for m in timed:
        if prev_ts is not None and (m["ts"] - prev_ts) > gap_seconds:
            if cur:
                sessions.append(cur)
            cur = []
        cur.append(m)
        prev_ts = m["ts"]
    if cur:
        sessions.append(cur)

    # untimed messages (if any) become their own trailing session, in file order
    if untimed:
        sessions.append(untimed)

    return sessions


def build_text(info_block: str, context_msgs: list[dict], target_msg: str,
               open_ctx: str, close_ctx: str, open_next: str, close_next: str
               ) -> tuple[str, str, str]:
    """Return (full_text, prompt, completion)."""
    ctx_lines = "\n".join(f'{m["user"]}: {m["msg"]}' for m in context_msgs)
    prompt = (
        f"{info_block}\n"
        f"{open_ctx}\n{ctx_lines}\n{close_ctx}\n"
        f"{open_next}"
    )
    completion = f"{target_msg}{close_next}"
    return prompt + completion, prompt, completion


def generate_windows(sessions: list[list[dict]], window: int, info_block: str,
                     delimiters: dict) -> list[dict]:
    samples = []
    for sess in sessions:
        for i, rec in enumerate(sess):
            if not target_eligible(rec):
                continue
            if i == 0:
                continue  # need at least one context message
            context = sess[max(0, i - window):i]
            if not context:
                continue
            full, prompt, completion = build_text(
                info_block, context, rec["msg"].strip(),
                delimiters["open_ctx"], delimiters["close_ctx"],
                delimiters["open_next"], delimiters["close_next"],
            )
            samples.append({
                "text": full,
                "prompt": prompt,
                "completion": completion,
            })
    return samples


def generate_qa_samples(msgs: list[dict], qa_info_block: str, delimiters: dict,
                        window_s: int, require_tagback: bool) -> list[dict]:
    """Find 'X tagged Y, Y replied within window_s seconds' pairs and turn each
    into a Q&A training sample:

        <info> ...qa flavour... </info>
        <chat>
        asker: <question, @tags stripped>
        </chat>
        <next><answer, @tags stripped></next>

    Same delimiters as the main format, so it's one model; only the info block
    differs to flip the model into 'answer the question' mode.
    """
    timed = sorted((m for m in msgs if isinstance(m["ts"], (int, float))),
                   key=lambda m: m["ts"])
    n = len(timed)
    out = []
    for i, q in enumerate(timed):
        tagged = {t.lower() for t in MENTION_RE.findall(q["msg"])}
        tagged.discard(q["user"].lower())          # ignore self-tags
        if not tagged:
            continue
        # find the FIRST reply from any tagged user inside the time window
        for j in range(i + 1, n):
            if timed[j]["ts"] - q["ts"] > window_s:
                break
            a = timed[j]
            if a["user"].lower() not in tagged or a["user"] == q["user"]:
                continue
            # precision filter: the answerer must tag the asker back
            if require_tagback and ("@" + q["user"].lower()) not in a["msg"].lower():
                break
            q_text = strip_mentions(q["msg"])
            a_text = strip_mentions(a["msg"])
            if not q_text or not a_text:
                break
            if a_text.startswith("!"):             # answer is a bot command
                break
            if q_text.lower() == a_text.lower():   # echo / copypasta bounce
                break
            full, prompt, completion = build_text(
                qa_info_block, [{"user": q["user"], "msg": q_text}], a_text,
                delimiters["open_ctx"], delimiters["close_ctx"],
                delimiters["open_next"], delimiters["close_next"],
            )
            out.append({
                "text": full, "prompt": prompt, "completion": completion,
                "_ts": a["ts"],
            })
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="chat_logs.jsonl")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--window", type=int, default=32,
                    help="max context messages per sample (default 32)")
    ap.add_argument("--session-gap", type=int, default=600,
                    help="seconds; gaps larger than this start a new session (default 600)")
    ap.add_argument("--bot-name", default="chatbot",
                    help="the name your live bot will post under")
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="fraction of LAST sessions held out for eval (time-based, default 0.05)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-qa", action="store_true",
                    help="skip the tagged-reply Q&A samples (on by default)")
    ap.add_argument("--qa-window", type=int, default=30,
                    help="seconds; a tagged user's reply within this counts as an answer (default 30)")
    ap.add_argument("--qa-loose", action="store_true",
                    help="include Q&A pairs where the answer does NOT tag the asker back "
                         "(more data, lower precision)")
    args = ap.parse_args()

    random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- delimiters (MUST match your Ollama Modelfile at serve time) ----
    delimiters = {
        "open_ctx": "<chat>",
        "close_ctx": "</chat>",
        "open_next": "<next>",
        "close_next": "</next>",
    }
    info_block = (
        "<info>\n"
        f"you are a viewer in this twitch chat. your name is {args.bot_name}.\n"
        "chat is fast, casual, full of emotes and slang. write one short message.\n"
        "</info>"
    )
    # Different info block flips the model into "answer the question" mode.
    qa_info_block = (
        "<info>\n"
        f"you are a viewer in this twitch chat. your name is {args.bot_name}.\n"
        "someone in chat asked you something. answer them with one short message.\n"
        "</info>"
    )

    # ---- load & sessionise ----
    msgs = load_messages(Path(args.input))
    sessions = split_sessions(msgs, args.session_gap)

    print(f"Loaded {len(msgs)} messages")
    print(f"Split into {len(sessions)} sessions (gap > {args.session_gap}s)")
    sizes = [len(s) for s in sessions]
    print(f"Session sizes: min={min(sizes)} max={max(sizes)} "
          f"avg={sum(sizes)/len(sizes):.0f}")

    # ---- time-based train/val split ----
    # Hold out a contiguous tail of the TIMELINE for eval, not whole sessions
    # (session sizes are wildly uneven, and a random split would leak
    #  near-identical copypasta windows into eval). We generate all windows
    #  first, then cut the eval set from the chronologically-last messages.
    #
    # To keep eval windows clean we split at a target-message cutoff time:
    # any sample whose TARGET message occurs after the cutoff goes to val,
    # everything else to train. Context may reach slightly before the cutoff,
    # which is fine and realistic.

    # Build windows per session but tag each sample with its target timestamp.
    def generate_with_ts(sess_list):
        out = []
        for sess in sess_list:
            for i, rec in enumerate(sess):
                if i == 0 or not target_eligible(rec):
                    continue
                context = sess[max(0, i - args.window):i]
                if not context:
                    continue
                full, prompt, completion = build_text(
                    info_block, context, rec["msg"].strip(),
                    delimiters["open_ctx"], delimiters["close_ctx"],
                    delimiters["open_next"], delimiters["close_next"],
                )
                out.append({
                    "text": full, "prompt": prompt, "completion": completion,
                    "_ts": rec["ts"] if isinstance(rec["ts"], (int, float)) else None,
                })
        return out

    all_samples = generate_with_ts(sessions)
    n_chat = len(all_samples)

    # ---- tagged-reply Q&A samples (different info block, same delimiters) ----
    n_qa = 0
    if not args.no_qa:
        qa_samples = generate_qa_samples(
            msgs, qa_info_block, delimiters,
            window_s=args.qa_window, require_tagback=not args.qa_loose,
        )
        n_qa = len(qa_samples)
        all_samples.extend(qa_samples)
        print(f"Generated {n_chat} next-message samples + {n_qa} Q&A samples "
              f"(tagged reply within {args.qa_window}s"
              f"{'' if args.qa_loose else ', answerer tags back'})")

    if args.val_frac > 0:
        ts_vals = sorted(s["_ts"] for s in all_samples if s["_ts"] is not None)
        if ts_vals:
            cutoff_idx = int(len(ts_vals) * (1 - args.val_frac))
            cutoff_idx = min(cutoff_idx, len(ts_vals) - 1)
            cutoff_ts = ts_vals[cutoff_idx]
            train_samples = [s for s in all_samples
                             if s["_ts"] is None or s["_ts"] < cutoff_ts]
            val_samples = [s for s in all_samples
                           if s["_ts"] is not None and s["_ts"] >= cutoff_ts]
        else:
            train_samples, val_samples = all_samples, []
    else:
        train_samples, val_samples = all_samples, []

    # strip the internal _ts tag before writing
    for s in train_samples:
        s.pop("_ts", None)
    for s in val_samples:
        s.pop("_ts", None)

    # shuffle TRAIN only (val stays as-is, order irrelevant)
    random.shuffle(train_samples)

    # ---- write ----
    train_path = outdir / "train.jsonl"
    val_path = outdir / "val.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    if val_samples:
        with open(val_path, "w", encoding="utf-8") as f:
            for s in val_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ---- save the exact format spec so serve-time matches ----
    spec = {
        "window": args.window,
        "session_gap_seconds": args.session_gap,
        "bot_name": args.bot_name,
        "delimiters": delimiters,
        "info_block": info_block,
        "loss_mask": "completion only (mask out the prompt portion)",
        "note": "Modelfile TEMPLATE and stop tokens must match these delimiters byte-for-byte.",
    }
    with open(outdir / "format_spec.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)

    # ---- report ----
    print("\n=== OUTPUT ===")
    print(f"train samples: {len(train_samples)}  -> {train_path}")
    print(f"val samples:   {len(val_samples)}  -> {val_path if val_samples else '(none)'}")
    print(f"format spec:   {outdir / 'format_spec.json'}")
    if train_samples:
        print("\n--- example training sample (text field) ---")
        print(train_samples[0]["text"])
        print("\n--- loss is computed ONLY on this completion portion ---")
        print(repr(train_samples[0]["completion"]))


if __name__ == "__main__":
    main()
