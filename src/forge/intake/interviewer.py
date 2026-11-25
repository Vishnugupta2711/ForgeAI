"""Clarifying intake interviewer supporting interactive, --answers, and --auto modes."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from forge.config import load_prompt
from forge.ingest.models import CorpusProfile
from forge.intake.models import (
    CapabilityNeed,
    IntakeAnswer,
    IntakeQuestion,
    RefusalTopic,
    RequirementsDocument,
    RequirementTopic,
)
from forge.llm.client import LLMClient

MAX_INTAKE_QUESTIONS = 5


class _QuestionList(BaseModel):
    questions: list[IntakeQuestion] = Field(default_factory=list)


class _AutoAnswerItem(BaseModel):
    question_id: str
    answer: str


class _AutoAnswerList(BaseModel):
    answers: list[_AutoAnswerItem] = Field(default_factory=list)


class _SynthesizedRequirements(BaseModel):
    scope: str = "Answer questions strictly grounded in the supplied documents."
    audience: str = "General users consulting the supplied documentation."
    tone: str = "Concise, factual, and helpful."
    must_answer_topics: list[RequirementTopic] = Field(default_factory=list)
    must_refuse_topics: list[RefusalTopic] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    capabilities_needed: list[CapabilityNeed] = Field(default_factory=list)


def _load_answers_file(path: Path) -> dict[str, str] | list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Answers file not found: {path}")
    raw_text = path.read_text(encoding="utf-8")
    loaded = (
        json.loads(raw_text)
        if path.suffix.lower() == ".json"
        else yaml.safe_load(raw_text)
    )
    if isinstance(loaded, list):
        return [str(x) for x in loaded]
    if isinstance(loaded, dict):
        if "answers" in loaded and isinstance(loaded["answers"], list):
            items = loaded["answers"]
            if all(isinstance(x, dict) for x in items):
                return {
                    str(x.get("question_id") or x.get("id") or f"Q{i + 1}"): str(
                        x.get("answer", "")
                    )
                    for i, x in enumerate(items)
                }
            return [str(x) for x in items]
        return {str(k): str(v) for k, v in loaded.items()}
    raise ValueError("Answers file must contain a YAML/JSON mapping or list.")


class IntakeInterviewer:
    """Runs intake questioning and synthesizes a structured RequirementsDocument."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def generate_questions(
        self, brief: str, profile: CorpusProfile
    ) -> list[IntakeQuestion]:
        """Generate clarifying questions and strictly cap at MAX_INTAKE_QUESTIONS (5)."""
        prompt = load_prompt("intake_questions.v1.md").format(
            brief=brief.strip(),
            corpus_profile=profile.to_compact_text(),
        )
        try:
            parsed = self.llm_client.json(
                prompt,
                schema=_QuestionList,
                role="agent",
                max_repairs=2,
            )
            raw_questions = parsed.questions
        except Exception:
            raw_questions = []

        # Enforce <= 5 cap IN CODE regardless of model output
        capped: list[IntakeQuestion] = []
        for idx, q in enumerate(raw_questions[:MAX_INTAKE_QUESTIONS], start=1):
            text = q.question.strip()
            if not text:
                continue
            capped.append(
                IntakeQuestion(
                    id=f"Q{idx}",
                    question=text,
                    why=q.why.strip(),
                )
            )
        return capped

    def collect_answers(
        self,
        questions: list[IntakeQuestion],
        brief: str,
        profile: CorpusProfile,
        *,
        mode: Literal["interactive", "answers", "auto"] = "auto",
        answers_path: Path | None = None,
        input_fn: Callable[[str], str] | None = None,
    ) -> tuple[list[IntakeAnswer], list[str]]:
        """Collect answers via interactive CLI, answers file, or simulated requester (--auto)."""
        if not questions:
            return [], []

        collected: list[IntakeAnswer] = []
        auto_assumptions: list[str] = []

        if mode == "answers":
            if answers_path is None:
                raise ValueError("answers_path is required when mode='answers'.")
            loaded = _load_answers_file(answers_path)
            for idx, q in enumerate(questions):
                if isinstance(loaded, list):
                    ans = loaded[idx].strip() if idx < len(loaded) else ""
                else:
                    ans = (loaded.get(q.id) or loaded.get(q.question) or "").strip()
                if not ans:
                    ans = "No answer provided; rely strictly on brief and documents."
                collected.append(
                    IntakeAnswer(
                        question_id=q.id,
                        question=q.question,
                        answer=ans,
                        source="answers_file",
                    )
                )
            return collected, auto_assumptions

        if mode == "interactive":
            prompt_reader = input_fn or (lambda prompt_str: input(prompt_str))
            for q in questions:
                try:
                    raw = prompt_reader(f"[{q.id}] {q.question}\n> ")
                except EOFError:
                    raw = ""
                ans = raw.strip() or "No preference specified; use conservative default."
                collected.append(
                    IntakeAnswer(
                        question_id=q.id,
                        question=q.question,
                        answer=ans,
                        source="interactive",
                    )
                )
            return collected, auto_assumptions

        # mode == "auto"
        q_block = "\n".join(f"- {q.id}: {q.question}" for q in questions)
        prompt = load_prompt("intake_auto_answer.v1.md").format(
            brief=brief.strip(),
            corpus_profile=profile.to_compact_text(),
            questions_block=q_block,
        )
        by_id: dict[str, str] = {}
        try:
            parsed = self.llm_client.json(
                prompt,
                schema=_AutoAnswerList,
                role="agent",
                max_repairs=2,
            )
            for item in parsed.answers:
                by_id[item.question_id.strip()] = item.answer.strip()
        except Exception:
            by_id = {}

        for q in questions:
            ans = by_id.get(q.id, "").strip()
            if not ans:
                ans = (
                    "Not specified in brief or corpus; assume strict grounding in "
                    "supplied documents and polite refusal of unsupported requests."
                )
            collected.append(
                IntakeAnswer(
                    question_id=q.id,
                    question=q.question,
                    answer=ans,
                    source="auto",
                )
            )
            auto_assumptions.append(f"Auto-assumed ({q.id}: {q.question}): {ans}")

        return collected, auto_assumptions

    def synthesize(
        self,
        brief: str,
        profile: CorpusProfile,
        qa_list: list[IntakeAnswer],
        auto_assumptions: list[str],
    ) -> RequirementsDocument:
        """Synthesize structured RequirementsDocument from brief, profile, and Q&A."""
        qa_block = "\n".join(
            f"- {qa.question_id} ({qa.source}): {qa.question} -> {qa.answer}"
            for qa in qa_list
        ) or "(none)"
        prompt = load_prompt("intake_synthesize.v1.md").format(
            brief=brief.strip(),
            corpus_profile=profile.to_compact_text(),
            qa_block=qa_block,
        )

        try:
            synth = self.llm_client.json(
                prompt,
                schema=_SynthesizedRequirements,
                role="agent",
                max_repairs=2,
            )
        except Exception:
            synth = _SynthesizedRequirements(
                scope=f"Answer questions grounded in the provided documents for: {brief.strip()}",
                must_answer_topics=[
                    RequirementTopic(
                        id="REQ-001",
                        title="Primary Brief Coverage",
                        description=brief.strip(),
                        priority="high",
                    )
                ],
            )

        normalized_topics: list[RequirementTopic] = []
        for idx, topic in enumerate(synth.must_answer_topics, start=1):
            req_id = (
                topic.id.strip()
                if topic.id.strip().startswith("REQ-")
                else f"REQ-{idx:03d}"
            )
            normalized_topics.append(
                RequirementTopic(
                    id=req_id,
                    title=topic.title.strip(),
                    description=topic.description.strip(),
                    priority=topic.priority,
                )
            )

        if not normalized_topics:
            normalized_topics.append(
                RequirementTopic(
                    id="REQ-001",
                    title="Primary Brief Coverage",
                    description=brief.strip(),
                    priority="high",
                )
            )

        merged_assumptions: list[str] = []
        for item in [*synth.assumptions, *auto_assumptions]:
            cleaned = item.strip()
            if cleaned and cleaned not in merged_assumptions:
                merged_assumptions.append(cleaned)

        return RequirementsDocument(
            brief=brief.strip(),
            scope=synth.scope.strip(),
            audience=synth.audience.strip(),
            tone=synth.tone.strip(),
            must_answer_topics=normalized_topics,
            must_refuse_topics=list(synth.must_refuse_topics),
            assumptions=merged_assumptions,
            capabilities_needed=list(synth.capabilities_needed),
            clarifying_qa=list(qa_list),
        )

    def run(
        self,
        brief: str,
        profile: CorpusProfile,
        *,
        mode: Literal["interactive", "answers", "auto"] = "auto",
        answers_path: Path | None = None,
        input_fn: Callable[[str], str] | None = None,
    ) -> RequirementsDocument:
        """Run full intake interview: questions -> answers -> synthesized requirements."""
        questions = self.generate_questions(brief, profile)
        qa_list, auto_assumptions = self.collect_answers(
            questions,
            brief,
            profile,
            mode=mode,
            answers_path=answers_path,
            input_fn=input_fn,
        )
        return self.synthesize(brief, profile, qa_list, auto_assumptions)
