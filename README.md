# GAIA Agent — Hugging Face Agents Course

I built this agent while working through Hugging Face's Agents Course, using
[smolagents](https://github.com/huggingface/smolagents) and a local LLM (Ollama, no API
costs). The final test was answering a subset of the
[GAIA benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard) — a set of
questions that sound simple but actually require real reasoning and tool use to answer
correctly. I scored 30% (6/20), which earned the Certificate of Completion. I also passed
Unit 1 with a perfect quiz score for the Certificate of Fundamentals.

**Certificate verification / agent code:** [Final_Assignment_Template on Hugging Face](https://huggingface.co/spaces/misialyna/Final_Assignment_Template)

## How the agent works

smolagents uses a "code agent" design: instead of the model picking from a fixed menu of
tool calls, it writes actual Python that gets executed — closer to how a person would
script a solution than to a typical chatbot tool-call loop.

I added a few tools on top of the basics:

- **`transcribe_audio`** — runs Whisper locally on any attached audio file
- **`read_excel_file`** — reads spreadsheet attachments with pandas
- **`wikipedia_lookup`** — pulls a clean Wikipedia summary instead of noisy search
  snippets, which turned out to matter a lot for factual questions
- **`check_botanical_classification`** — a small reference table so the agent can tell
  botanical fruits from vegetables reliably instead of guessing
- **`get_original_question_text` / `get_reversed_question_text`** — these exist because
  of a bug I kept running into: when a question needed exact string manipulation, the
  model would sometimes retype it from memory and silently drop part of it. Giving it a
  tool that returns the text programmatically fixed that completely.

## What actually went wrong along the way

I think the failures are more interesting than the successes, so I kept them in:

Plain web search wasn't precise enough for questions that needed a specific fact from a
specific source (like counting an artist's studio albums in a date range) — the agent
kept getting fragments instead of answers. Adding the Wikipedia tool fixed most of that.

I also tried a manager/worker multi-agent setup for one task, and the local model really
struggled with it — one run took 13 steps and around 56k tokens to land on an answer that
was correct but clearly fragile. Single-tool tasks needed 1-3 steps. Smaller local models
and multi-agent coordination don't seem to mix well yet.

The most important lesson was about verification. Twice, when the agent couldn't find
real information, it answered anyway with something that sounded confident but had zero
supporting evidence — a country name pulled out of nowhere, a person from the wrong
decade entirely. I caught both by checking the agent's own reasoning trace against
independent sources before submitting anything. Neither answer was used. A certificate is
only worth something if it reflects what the agent can actually do, so I'd rather submit
fewer answers I've verified than more answers I'm not sure about.

I also learned that GAIA's exact-match scoring is genuinely unforgiving — "St.
Petersburg" scored differently than "Saint Petersburg" for one question, purely because
the question asked for no abbreviations.

## Files

- `gaia_agent.py` — the agent, its tools, and the model configuration

## Stack

Python, smolagents, Ollama, Whisper, pandas, Wikipedia API.
