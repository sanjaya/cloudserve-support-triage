# Prompt register

| ID | Name | Category | Serves | Version | File |
|---|---|---|---|---|---|
| PR-01 | Requirement to specification | Specification | FR-01 to FR-12 | 1.0 | specification/PR-01_requirement_to_spec_v1.0.md |
| PR-02 | Intent and urgency classifier | Build | FR-02, FR-10 | 1.0 | build/PR-02_classify_intent_urgency_v1.0.md |
| PR-03 | Grounded answer drafter | Build | FR-06, FR-07 | 1.0 | build/PR-03_grounded_answer_v1.0.md |
| PR-03 | Grounded answer drafter | Build | FR-06, FR-07 | 1.1 | build/PR-03_grounded_answer_v1.1.md |
| PR-04 | Escalation hand-off summary | Build | FR-08 | 1.0 | build/PR-04_escalation_summary_v1.0.md |
| PR-05 | Grounding and commitment check | Build (guardrail) | FR-07, NFR-03 | 1.0 | build/PR-05_grounding_check_v1.0.md |
| PR-06 | Instruction-integrity check | Build (guardrail) | FR-07 | 1.0 | build/PR-06_instruction_integrity_v1.0.md |
| PR-07 | Review against requirement | Review | FR-01 to FR-12 | 1.0 | review/PR-07_review_against_requirement_v1.0.md |
| PR-08 | Answer quality judge | Evaluation | FR-06, NFR-03, success measures | 1.0 | evaluation/PR-08_answer_quality_judge_v1.0.md |

Rules: a change to prompt text creates a new file with the next version number; the old file stays. Every decision-log record stores the prompt id and version used. `{placeholders}` are filled by code; customer text only ever goes inside the `<ticket>` tag.

## Change history

- PR-03 v1.1 (5 Nov 2026): defines when to use "answered", "partial" and "insufficient". In the
  first validation run, v1.0 said "partial" for 30 tickets the articles can answer, because it
  could not explain the exact cause in the customer's account. v1.1 also asks for line breaks
  as \n. Chosen between v1.0 and v1.1 on development tickets with scripts/evaluate_answers.py.
