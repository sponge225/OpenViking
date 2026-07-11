from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger


EVIDENCE_BASED_ASSESSMENT_INSTRUCTION = """IMPORTANT: Answer strictly based on the provided context above. Do NOT use external knowledge or information not present in the context.

Before answering, perform a strict sufficiency audit. Context relevance is not enough: the answer is sufficient only when every required part of the question is directly supported by the provided context.

Audit steps:
1. Question requirements: identify the exact answer type requested and every constraint in the question, such as entity, version, date, count, scope, comparison target, yes/no condition, or multi-hop link.
2. Direct support: for each required part, quote the exact sentence(s) from the context that support it. If support requires combining multiple quoted facts, explain that combination briefly.
3. Unsupported or inferred parts: list any required part that is missing, only weakly related, only implied, contradicted, ambiguous, from a similar-but-different entity/scope/version, or would require guessing.

Routing decision:
- Set "action" to "answer" only when every required part is directly supported by quoted context.
- Set "action" to "fallback" when any required part is missing, weakly supported, inferred, ambiguous, conflicting, or only supported by related-but-not-exact context. In that case, set "sufficient" to false, list the missing or unsupported parts in "missing_info", and set "answer" to "Not mentioned". Do NOT guess or fabricate.
- If the question requires a number, date, version, name, location, yes/no conclusion, or comparison result, that exact value or conclusion must be directly supported by the quoted context.

If the context is SUFFICIENT, set "action" to "answer", set "sufficient" to true, and provide a complete answer in the "answer" field. Include all relevant details rather than oversimplifying.

Respond ONLY as a JSON object in the following format:
{
  "action": "answer" | "fallback",
  "sufficient": true/false,
  "evidence_analysis": [
    "Question requirements: ...",
    "Direct support: [Quote] ...",
    "Unsupported or inferred parts: ..."
  ],
  "missing_info": [],
  "answer": "<final answer or Not mentioned>",
  "reasoning": "<one short sentence summarizing why every requirement is supported or what is unsupported>"
}"""


SIMPLE_CONTEXT_ANSWER_INSTRUCTION = """Answer the question using only the provided context.

This is a simple baseline prompt. Do not perform a detailed evidence audit.
If the context appears to contain enough relevant information, set "sufficient" to true and provide the answer.
If the context does not contain enough relevant information, set "sufficient" to false and set "answer" to "Not mentioned".

Return JSON only:
{
  "sufficient": true/false,
  "answer": "<final answer or Not mentioned>",
  "reasoning": "<one short sentence>"
}"""


@dataclass
class StandardQA:
    """Standardized single question-answer pair"""
    question: str
    gold_answers: List[str]
    evidence: List[str] = field(default_factory=list)
    category: Optional[Union[int, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StandardSample:
    """Standardized sample containing document content and corresponding QA list"""
    sample_id: str
    qa_pairs: List[StandardQA]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class StandardDoc:
    """Standardized sampleid to doc_path mapping structure"""
    sample_id:str
    doc_path:str


class BaseAdapter(ABC):
    """Base class for all dataset adapters"""
    
    def __init__(self, raw_file_path: str):
        self.raw_file_path = raw_file_path
        self.logger = get_logger()

    @abstractmethod
    def data_prepare(self, doc_dir:str) -> List[StandardDoc]:
        """
        Data preparation.
        1. Convert dataset format to OpenViking-friendly format
        2. Return converted (or unconverted) file paths
        
        Returns:
            List[StandardDoc]: Array of file paths expected to be input to OpenViking
        """
        pass

    @abstractmethod
    def load_and_transform(self) -> List[StandardSample]:
        """
        Read raw files and convert to standard format list.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> tuple[str, Dict[str, Any]]:
        """
        Build final prompt to send to LLM based on retrieved context and QA pair.
        
        Returns:
            - full_prompt (str): Complete prompt string
            - meta (Dict): Metadata to pass to post-processing function (e.g., option mapping for multiple choice)
        """
        pass

    def post_process_answer(self, qa: StandardQA, raw_answer: str, meta: Dict[str, Any]) -> str:
        """
        Post-process raw LLM output (default implementation only strips whitespace).
        """
        return raw_answer.strip()

    def build_simple_prompt(self, qa: StandardQA, context_blocks: List[str]) -> tuple[str, Dict[str, Any]]:
        """
        Build a deliberately simple QA prompt for baseline fallback providers.
        This is separate from build_prompt so the main adapter prompt can keep
        its richer dataset-specific behavior.
        """
        context_text = "\n\n".join(str(block) for block in context_blocks)
        full_prompt = f"{context_text}\n\n{SIMPLE_CONTEXT_ANSWER_INSTRUCTION}\n\nQuestion: {qa.question}"
        return full_prompt, {}

    def evidence_selection_instruction(self, qa: StandardQA) -> str:
        """
        Optional dataset-specific guidance for selecting evidence from retrieved context.
        Subclasses may override this to add domain constraints without changing the
        shared pipeline prompt.
        """
        return ""

    def evidence_sufficiency_instruction(self, qa: StandardQA) -> str:
        """
        Optional dataset-specific guidance for judging whether selected evidence is
        sufficient to answer the question.
        """
        return ""
