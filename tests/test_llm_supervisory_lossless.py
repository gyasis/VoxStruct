"""Tests for LLMSupervisor: lossless headers, fact-checked report mode, key handling (no network)."""
import json
import re
from types import SimpleNamespace

import voxstruct.utils.llm_supervisory as ls
from voxstruct.utils.llm_supervisory import LLMSupervisor


def _reply(content, finish_reason="stop"):
    msg = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)])


def _sup(**kw):
    return LLMSupervisor(api_key="x", model="openai/test", **kw)


PARAS = [
    "[00:00:01] Welcome to the show, today we talk about loops.",
    "[00:00:46] Looped transformers reuse the same layers several times.",
    "[00:01:31] Now the hardware: you need sixteen gigabytes of RAM.",
    "[00:02:16] That is all, thanks for watching.",
]


def test_structure_lossless_keeps_every_paragraph(monkeypatch):
    sections = [{"index": 0, "title": "Intro"}, {"index": 2, "title": "Hardware Needs"}]
    monkeypatch.setattr(ls, "completion", lambda **k: _reply(json.dumps(sections)))
    md, secs = _sup().structure_lossless(PARAS)
    assert md.split("\n\n") == ["## Intro", PARAS[0], PARAS[1], "## Hardware Needs", PARAS[2], PARAS[3]]
    assert secs == sections


def test_structure_lossless_ignores_bad_indices_and_bad_json(monkeypatch):
    monkeypatch.setattr(ls, "completion",
                        lambda **k: _reply('```json\n[{"index": 99, "title": "Ghost"}, {"index": 1, "title": "Loops"}]\n```'))
    md, secs = _sup().structure_lossless(PARAS)
    assert secs == [{"index": 0, "title": "Introduction"}, {"index": 1, "title": "Loops"}]
    assert [p for p in md.split("\n\n") if not p.startswith("## ")] == PARAS

    monkeypatch.setattr(ls, "completion", lambda **k: _reply("not json"))
    md, secs = _sup().structure_lossless(PARAS)
    assert secs == [{"index": 0, "title": "Introduction"}]
    assert md == "## Introduction\n\n" + "\n\n".join(PARAS)


def test_structure_lossless_windows_keep_global_indices(monkeypatch):
    seen = []

    def fake(**k):
        prompt = k["messages"][0]["content"]
        first = int(re.search(r"<(\d+)> ", prompt).group(1))
        seen.append(first)
        return _reply(json.dumps([{"index": first, "title": f"Part {first}"}]))

    monkeypatch.setattr(ls, "completion", fake)
    md, secs = _sup().structure_lossless(PARAS, window_chars=120)
    assert len(seen) > 1
    assert [s["index"] for s in secs] == seen
    assert [p for p in md.split("\n\n") if not p.startswith("## ")] == PARAS


def test_short_rewrite_is_not_rejected_for_word_count(monkeypatch):
    raw = "um so uh welcome to the show today we we talk about loops"
    monkeypatch.setattr(ls, "completion", lambda **k: _reply("Topic: loops."))
    assert _sup().validate_and_improve_transcript(raw) == "Topic: loops."


def test_truncated_rewrite_is_rejected(monkeypatch):
    raw = "welcome to the show today we talk about loops"
    monkeypatch.setattr(ls, "completion", lambda **k: _reply(raw + " extra", finish_reason="length"))
    assert _sup().validate_and_improve_transcript(raw) == raw


def test_focus_mode_may_drop_content(monkeypatch):
    raw = " ".join(PARAS)
    monkeypatch.setattr(ls, "completion", lambda **k: _reply("Hardware: sixteen gigabytes of RAM."))
    out = _sup().validate_and_improve_transcript(raw, focus="hardware")
    assert out == "Hardware: sixteen gigabytes of RAM."


def test_completion_kwargs_are_passed_and_no_output_cap(monkeypatch):
    got = {}

    def fake(**k):
        got.update(k)
        return _reply("welcome")

    monkeypatch.setattr(ls, "completion", fake)
    _sup(completion_kwargs={"reasoning_effort": "none"}).validate_and_improve_transcript("welcome")
    assert got["reasoning_effort"] == "none"
    assert "max_tokens" not in got


def test_key_is_passed_per_call_and_env_untouched(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
    got = {}

    def fake(**k):
        got.update(k)
        return _reply("welcome")

    monkeypatch.setattr(ls, "completion", fake)
    LLMSupervisor(api_key="gateway-key", model="openai/test").validate_and_improve_transcript("welcome")
    import os
    assert os.environ["OPENAI_API_KEY"] == "real-openai-key"
    assert got["api_key"] == "gateway-key"


def test_facts_and_missing_facts():
    src = "[00:01:00] The RAON-1 Nano has 1.5b params, runs on an RTX 5090 at 85.9 tokens, uses gpt-4o and OpenBot."
    f = LLMSupervisor.facts(src)
    assert "1.5" in f["numbers"] and "5090" in f["numbers"] and "85.9" in f["numbers"]
    assert "00" not in f["numbers"]
    assert {"RAON-1", "RTX", "gpt-4o", "OpenBot"} <= set(f["terms"])
    good = "RAON-1 Nano: 1.5 billion parameters. Runs on an rtx 5090 at 85.9 tokens/s. Uses GPT-4o and OpenBot."
    assert LLMSupervisor.missing_facts(src, good) == []
    assert LLMSupervisor.missing_facts(src, "RAON-1 Nano is small. Uses gpt-4o, OpenBot, RTX.") == ["1.5", "5090", "85.9"]


def _sections_then(*drafts):
    """Fake completion: first call returns sections, then the given drafts in order."""
    calls = []

    def fake(**k):
        calls.append(k["messages"][0]["content"])
        if len(calls) == 1:
            return _reply(json.dumps([{"index": 0, "title": "Intro"}, {"index": 2, "title": "Hardware"}]))
        return _reply(drafts[len(calls) - 2])
    return fake, calls


def test_structure_report_retries_then_accepts(monkeypatch):
    fake, calls = _sections_then(
        "- Topic: looped transformers reuse layers.",
        "- Hardware: needs RAM.",
        "- Hardware: needs 16 GB of RAM.",
    )
    monkeypatch.setattr(ls, "completion", fake)
    paras = [PARAS[0], PARAS[1], "[00:01:31] Now the hardware: you need 16 GB of RAM.", "[00:02:16] Thanks."]
    md, rep = _sup().structure_report(paras)
    assert md.startswith("## Intro [00:00:01]\n\n- Topic")
    assert "## Hardware [00:01:31]\n\n- Hardware: needs 16 GB" in md
    assert rep[1]["attempts"] == 2 and rep[1]["verbatim"] is False
    assert "left out: 16" in calls[3]


def test_structure_report_keeps_source_when_facts_stay_missing(monkeypatch):
    fake, _ = _sections_then("- Intro.", "- Hardware.", "- Hardware.")
    monkeypatch.setattr(ls, "completion", fake)
    paras = [PARAS[0], PARAS[1], "[00:01:31] You need 16 GB of RAM.", "[00:02:16] Thanks."]
    md, rep = _sup().structure_report(paras)
    assert rep[1]["verbatim"] is True and "16" in rep[1]["missing"]
    assert "## Hardware [00:01:31]\n\n[00:01:31] You need 16 GB of RAM.\n\n[00:02:16] Thanks." in md


def test_number_values_reads_spoken_numbers():
    v = LLMSupervisor._number_values(
        "six point three points, one point six times, twelve and a half, 12 and a half, "
        "three hundred and fifty configs, twenty one experts, 1.5b, 2 million")
    assert {"6.3", "1.6", "12.5", "12", "350", "21", "1.5", "2"} <= v


def test_added_numbers_flags_invented_year():
    src = "This year it is split between Sydney and Europe. Training is one point six times faster."
    out = "The 2024 event is split between Sydney and Europe. Training is 1.6x faster."
    assert LLMSupervisor.added_numbers(src, out) == ["2024"]


def test_structure_report_rejects_invented_number(monkeypatch):
    fake, calls = _sections_then("- Intro.", "- Hardware: 16 GB of RAM, released in 2024.",
                                 "- Hardware: 16 GB of RAM.")
    monkeypatch.setattr(ls, "completion", fake)
    paras = [PARAS[0], PARAS[1], "[00:01:31] You need 16 GB of RAM.", "[00:02:16] Thanks."]
    md, rep = _sup().structure_report(paras)
    assert rep[1]["attempts"] == 2 and not rep[1]["verbatim"]
    assert "not in the transcript: 2024" in calls[3]
    assert "2024" not in md


def test_and_a_half_is_kept_by_decimal():
    src = "Mini finished in 8 and a half minutes; the old one scores just 12 and a half."
    assert LLMSupervisor.missing_facts(src, "Mini finished in 8.5 minutes. The old model scores 12.5.") == []
    assert LLMSupervisor.missing_facts(src, "Mini finished in 9 minutes. The old model scores 12.5.") == ["8"]


def test_article_has_title_standfirst_and_outline_context(monkeypatch):
    fake, calls = _sections_then("Intro prose.", "Hardware prose with 16 GB of RAM.",
                                 "This talk covers loops and needs 16 GB of RAM.")
    monkeypatch.setattr(ls, "completion", fake)
    paras = [PARAS[0], PARAS[1], "[00:01:31] You need 16 GB of RAM.", "[00:02:16] Thanks."]
    md, rep = _sup().structure_report(paras, title="Loops Talk")
    assert md.startswith("# Loops Talk\n\n*This talk covers loops and needs 16 GB of RAM.*\n\n## Intro [00:00:01]")
    assert "Sections in order: Intro | Hardware" in calls[1] and "Write the section: Hardware" in calls[2]
    assert "magazine feature" in calls[1]


def test_opening_with_invented_number_is_dropped(monkeypatch):
    fake, _ = _sections_then("Intro prose.", "Hardware prose with 16 GB of RAM.",
                             "Released in 2024, this talk needs 16 GB.")
    monkeypatch.setattr(ls, "completion", fake)
    paras = [PARAS[0], PARAS[1], "[00:01:31] You need 16 GB of RAM.", "[00:02:16] Thanks."]
    md, _ = _sup().structure_report(paras, title="Loops Talk")
    assert md.startswith("# Loops Talk\n\n## Intro") and "2024" not in md


def test_unfaithful_quotes():
    src = "[00:00:01] I think looped models are the future of local AI, honestly."
    assert LLMSupervisor.unfaithful_quotes(src, 'The speaker says "looped models are the future of local AI".') == []
    assert LLMSupervisor.unfaithful_quotes(src, 'The speaker says "loops will replace every model soon".') == \
        ["loops will replace every model soon"]


def test_paper_style_regroups_with_abstract_and_footer(monkeypatch):
    calls = []
    replies = [json.dumps({"Background": [0], "Approach": [1], "Findings": [2]}),
               "Background prose.", "Approach prose.", "Findings: 16 GB of RAM.",
               "Limitations prose.", "Abstract prose."]

    def fake(**k):
        calls.append(k["messages"][0]["content"])
        return _reply(replies[len(calls) - 1])

    monkeypatch.setattr(ls, "completion", fake)
    paras = [PARAS[0], PARAS[1], "[00:01:31] You need 16 GB of RAM.", "[00:02:16] Thanks."]
    md, rep = _sup().structure_report(paras, title="Loops", style="paper")
    names = [r["title"] for r in rep]
    assert names == ["Background", "Approach", "Findings", "Limitations"]  # unclaimed paragraph 3 -> last section
    assert sorted(i for r in rep for i in r["paragraphs"]) == [0, 1, 2, 3]
    assert md.startswith("# Loops\n\n## Abstract\n\nAbstract prose.")
    assert "*Transcript: [00:01:31]*" in md
    assert "research paper" in calls[1]


def test_news_style_ranks_sections(monkeypatch):
    calls = []
    replies = [json.dumps([{"index": 0, "title": "Intro"}, {"index": 2, "title": "Hardware"}]),
               "[1, 0]", "Hardware: 16 GB of RAM.", "Intro prose.", "Lede prose."]

    def fake(**k):
        calls.append(k["messages"][0]["content"])
        return _reply(replies[len(calls) - 1])

    monkeypatch.setattr(ls, "completion", fake)
    paras = [PARAS[0], PARAS[1], "[00:01:31] You need 16 GB of RAM.", "[00:02:16] Thanks."]
    md, rep = _sup().structure_report(paras, title="Loops", style="news")
    assert [r["title"] for r in rep] == ["Hardware", "Intro"]
    assert md.startswith("# Loops\n\nLede prose.\n\n## Hardware [00:01:31]")


def test_unknown_style_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        _sup().structure_report(PARAS, style="poem")


def test_quotes_pair_in_order():
    src = "[00:00:01] It is fast. Honestly the nano model is the one I would ship to players."
    out = 'He calls it "fast" and adds that "the nano model is the one I would ship to players".'
    assert LLMSupervisor.unfaithful_quotes(src, out) == []


def test_large_section_is_split_into_subsections(monkeypatch):
    calls = []
    big = ["[00:00:%02d] " % i + ("word " * 60) + f"value {i}" for i in range(6)]
    replies = [json.dumps({"Details": [0, 1, 2, 3, 4, 5]}),
               json.dumps([{"index": 0, "title": "First"}, {"index": 3, "title": "Second"}]),
               "First part: value 0, value 1, value 2.", "Second part: value 3, value 4, value 5.",
               "Summary prose."]

    def fake(**k):
        calls.append(k["messages"][0]["content"])
        return _reply(replies[len(calls) - 1])

    monkeypatch.setattr(ls, "completion", fake)
    md, rep = _sup().structure_report(big, title="T", style="report", section_chars=800)
    assert "## Details\n\n### First\n\nFirst part" in md and "### Second\n\nSecond part" in md
    assert [r["paragraphs"] for r in rep] == [[0, 1, 2], [3, 4, 5]]


def test_number_words_in_output_are_not_invented_numbers():
    src = "The nano model is small. It runs at 85.9 tokens."
    out = "The nano model is the one to pick. One reason: it runs at 85.9 tokens per second."
    assert LLMSupervisor.added_numbers(src, out) == []


def test_numbered_list_markers_are_not_invented_numbers():
    src = "First score every expert, then group them, then glue each group, then distill."
    out = "The process has four steps:\n1. Score every expert.\n2. Group them.\n3. Glue each group.\n4. Distill."
    assert LLMSupervisor.added_numbers(src, out) == []


def test_auto_style_uses_choice(monkeypatch):
    calls = []
    replies = ["Paper.", json.dumps({"Background": [0, 1], "Findings": [2, 3]}),
               "Background prose.", "Findings: 16 GB of RAM.", "Abstract prose."]

    def fake(**k):
        calls.append(k["messages"][0]["content"])
        return _reply(replies[len(calls) - 1])

    monkeypatch.setattr(ls, "completion", fake)
    paras = [PARAS[0], PARAS[1], "[00:01:31] You need 16 GB of RAM.", "[00:02:16] Thanks."]
    md, rep = _sup().structure_report(paras, title="T", style="auto")
    assert "## Abstract" in md and {r["style"] for r in rep} == {"paper"}


def test_choose_style_falls_back_to_article(monkeypatch):
    monkeypatch.setattr(ls, "completion", lambda **k: _reply("A podcast transcript."))
    assert _sup().choose_style(PARAS, "T") == "article"
