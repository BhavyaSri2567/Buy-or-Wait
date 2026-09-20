# Token Usage and Cost Report

Run produced: `C:\Users\kvdbh\OneDrive\Documents\Orchestrate Hackathon\hackerrank-orchestrate-september26\output.csv`

Requests processed: 250

Wall-clock runtime: 4.0s

## Model Usage

- Provider: Anthropic
- Model: claude-haiku-4-5-20251001
- Enabled this run: False
- Model calls: 0
- Input tokens: 0
- Output tokens: 0
- Total tokens: 0
- Avg tokens / call: 0.0
- Estimated cost: $0.0000 (placeholder rates, edit to your real pricing)

## Notes

The deterministic engine (currency conversion, recurrence detection, 90-day balance simulation, plan ranking) makes no model calls. The optional Anthropic API is used only to (a) read amounts off images linked to blank-amount financial events, and (b) interpret any message that does not match a known template. With ANTHROPIC_API_KEY unset, both fall back to deterministic heuristics documented in engine/events.py and engine/extraction.py, and this run made zero model calls.

## Data-Resolution Notes (sample)

- event_3051: blank amount, image unresolved; used category median 6506.01 as conservative estimate
- event_3231: blank amount, image unresolved; used category median 4608.50 as conservative estimate
- event_4535: blank amount unresolved, no history; treated as 0 (excluded)
- event_5170: blank amount, image unresolved; used category median 13529.34 as conservative estimate
- event_6033: blank amount, image unresolved; used category median 2549.55 as conservative estimate
- event_6859: blank amount unresolved, no history; treated as 0 (excluded)
- event_7307: blank amount, image unresolved; used category median 2169.89 as conservative estimate
- event_7941: blank amount, image unresolved; used category median 6011.00 as conservative estimate
- event_9421: blank amount, image unresolved; used category median 9509.61 as conservative estimate
- event_9806: blank amount, image unresolved; used category median 2013.41 as conservative estimate
- event_10521: blank amount, image unresolved; used category median 2255.64 as conservative estimate
