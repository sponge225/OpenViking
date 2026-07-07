import hashlib
import json
import os
import re
import threading
import time
from typing import Any


class QuestionRewriteError(RuntimeError):
    pass


_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _path_lock(path: str) -> threading.Lock:
    abs_path = os.path.abspath(path)
    with _PATH_LOCKS_GUARD:
        if abs_path not in _PATH_LOCKS:
            _PATH_LOCKS[abs_path] = threading.Lock()
        return _PATH_LOCKS[abs_path]


class QuestionRewriteStore:
    def __init__(self, output_dir: str, dataset_name: str = ""):
        self.path = os.path.join(output_dir, "question_rewrites.jsonl")
        self.dataset_name = dataset_name or ""
        self._lock = _path_lock(self.path)
        self._cache: dict[str, dict[str, Any]] | None = None

    @staticmethod
    def normalize_question(question: str) -> str:
        return re.sub(r"\s+", " ", (question or "").strip())

    def compute_id(self, sample_id: str, question: str) -> str:
        key = "\n".join([
            self.dataset_name,
            str(sample_id or ""),
            self.normalize_question(question).lower(),
        ])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._cache is not None:
            return self._cache

        cache: dict[str, dict[str, Any]] = {}
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rec_id = rec.get("id", "")
                    rewrites = rec.get("rewrites", [])
                    if rec_id and isinstance(rewrites, list) and len(rewrites) >= 2:
                        cache[rec_id] = rec

        self._cache = cache
        return cache

    def get(self, sample_id: str, question: str) -> list[str] | None:
        rec_id = self.compute_id(sample_id, question)
        with self._lock:
            self._cache = None
            rec = self._load().get(rec_id)
        if not rec:
            return None
        rewrites = rec.get("rewrites", [])
        if isinstance(rewrites, list) and len(rewrites) >= 2:
            return [str(x).strip() for x in rewrites[:2] if str(x).strip()]
        return None

    def put(self, sample_id: str, question: str, rewrites: list[str], model: str = "") -> None:
        rec_id = self.compute_id(sample_id, question)
        record = {
            "id": rec_id,
            "dataset": self.dataset_name,
            "sample_id": sample_id,
            "original_question": self.normalize_question(question),
            "rewrites": rewrites[:2],
            "model": model or "",
            "created_at": int(time.time()),
        }
        with self._lock:
            self._cache = None
            cache = self._load()
            if rec_id in cache:
                return
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            cache[rec_id] = record


def _extract_json_array(text: str) -> list[Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text or "").strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        raise QuestionRewriteError("rewrite response did not contain a JSON array")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise QuestionRewriteError(f"rewrite response JSON parse failed: {e}") from e
    if not isinstance(parsed, list):
        raise QuestionRewriteError("rewrite response was not a JSON array")
    return parsed


def parse_rewrites(raw: str, original_question: str) -> list[str]:
    parsed = _extract_json_array(raw)
    original_norm = QuestionRewriteStore.normalize_question(original_question).lower()
    rewrites: list[str] = []
    seen: set[str] = set()

    for item in parsed:
        if not isinstance(item, str):
            continue
        rewrite = QuestionRewriteStore.normalize_question(item)
        key = rewrite.lower()
        if not rewrite or key == original_norm or key in seen:
            continue
        seen.add(key)
        rewrites.append(rewrite)
        if len(rewrites) == 2:
            return rewrites

    raise QuestionRewriteError("rewrite response did not contain two distinct non-original questions")


def build_rewrite_prompt(question: str, previous_error: str = "", previous_output: str = "") -> str:
    retry_note = ""
    if previous_error:
        retry_note = (
            "\n\nThe previous response was invalid and will be discarded.\n"
            f"Validation error: {previous_error}\n"
        )
        if previous_output:
            retry_note += f"Invalid response excerpt: {previous_output[:500]}\n"
        retry_note += "Regenerate the answer and follow the JSON format exactly.\n"

    return f"""Rewrite the following question into exactly two semantically equivalent but differently phrased questions.

Rules:
- Preserve all entities, dates, objects, constraints, and intent.
- Do not add new facts, assumptions, filters, or requirements.
- Do not remove any key condition from the original question.
- Each rewrite must be different from the original and from each other.
- Output ONLY a JSON array of exactly two strings.
- The response must be parseable by json.loads().
{retry_note}

Question:
{question}

Output format:
["rewrite 1", "rewrite 2"]"""


def get_or_create_rewrites(
    store: QuestionRewriteStore,
    sample_id: str,
    question: str,
    llm: Any,
    model: str = "",
    max_attempts: int = 3,
) -> list[str]:
    cached = store.get(sample_id, question)
    if cached and len(cached) == 2:
        return cached

    if llm is None or not hasattr(llm, "generate"):
        raise QuestionRewriteError("LLM client does not support generate()")

    attempts = max(1, int(max_attempts or 1))
    last_error: Exception | None = None
    last_output = ""
    for attempt in range(1, attempts + 1):
        try:
            raw = llm.generate(
                build_rewrite_prompt(
                    question,
                    previous_error=str(last_error or ""),
                    previous_output=last_output,
                )
            )
            last_output = str(raw or "")
            rewrites = parse_rewrites(last_output, question)
            store.put(sample_id, question, rewrites, model=model)
            return rewrites
        except QuestionRewriteError as e:
            last_error = e
        except Exception as e:
            last_error = e
        if attempt < attempts:
            time.sleep(min(2.0, 0.25 * attempt))

    raise QuestionRewriteError(
        f"failed to generate valid question rewrites after {attempts} attempt(s): {last_error}"
    )
