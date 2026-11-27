# ForgeAI Demo Runs

Summary of end-to-end `forge run --auto` executions using local Ollama models (`forge.toml`). Metrics below will be populated strictly from the generated `runs/<name>/report.md` and `runs/<name>/scorecard.csv` artifacts once the live runs complete.

## Results Summary

| Demo | Agent Status | Dev Baseline | Dev Final | Held-Out Baseline | Held-Out Final (95% Wilson CI) | Tools Attached | Groundability Warnings | Runtime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `hr_policy` | Pending live run | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `ship_spares` | Pending live run | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 1. HR Policy Assistant (`demos/hr_policy`)

- **Brief**: [`demos/hr_policy/brief.txt`](hr_policy/brief.txt)
- **Corpus**: `leave_policy.pdf`, `code_of_conduct.pdf`, `expenses_and_travel_policy.pdf`
- **Artifacts**: `runs/hr_policy/` (pending live execution)
- **Chat Transcript**: [`demos/hr_policy/transcripts/`](hr_policy/transcripts/)

### Groundability & Tooling Notes

_To be populated from `runs/hr_policy/groundability_report.json` and `runs/hr_policy/report.md`._

### What Still Fails and Why

_To be populated verbatim from `runs/hr_policy/report.md`._

## 2. Ship-Spares Procurement Assistant (`demos/ship_spares`)

- **Brief**: [`demos/ship_spares/brief.txt`](ship_spares/brief.txt)
- **Corpus**: `spares_catalogue.csv` (160 rows), `supplier_terms_and_returns.pdf`, `logistics_and_lead_times.pdf`
- **Artifacts**: `runs/ship_spares/` (pending live execution)
- **Chat Transcript**: [`demos/ship_spares/transcripts/`](ship_spares/transcripts/)

### Groundability & Tooling Notes

_To be populated from `runs/ship_spares/groundability_report.json` and `runs/ship_spares/report.md`._

### What Still Fails and Why

_To be populated verbatim from `runs/ship_spares/report.md`._
