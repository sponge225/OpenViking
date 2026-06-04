from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger


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


ASSESSMENT_INSTRUCTION = """You are a strict answerability judge for a RAG system.

Your task is to decide whether the provided context contains enough information to answer the user question fully and accurately.

You must only use the provided context. Do not use outside knowledge. Do not infer facts that are not supported by the context.

If the question specifies a particular target scope (such as a specific document, product, time, entity, etc.), first verify that the provided context belongs to that target scope.
- If the context comes from other unrelated targets, do not use it.
- Only answer based on the context that belongs to the specified target scope.

A context is SUFFICIENT only if:
1. It directly contains the key facts needed to answer the question.
2. The answer can be produced without guessing, filling gaps, or relying on outside knowledge.
3. The relevant information is specific enough, not merely related or background information.
4. There is no unresolved conflict between different parts of the context.
5. The answer can address all important parts of the question, not just a subset.
6. For questions about a named target scope, the supporting context clearly belongs to that target.

A context is INSUFFICIENT if:
1. Key facts required by the question are missing.
2. The context only partially answers the question.
3. The context is related but does not directly support the answer.
4. The context contains conflicting information that cannot be resolved.
5. Answering would require assumptions, speculation, or external knowledge.
6. The context appears to come from a different target scope than the one specified in the question.

Be conservative. If you are not clearly confident that the context is sufficient, mark it as insufficient.

---

IF CONTEXT IS SUFFICIENT:
- Provide a concise and accurate answer based on the context.

IF CONTEXT IS INSUFFICIENT:
- Answer exactly: "Not mentioned"

Return your response in the following JSON format only:

{
  "sufficient": true/false,
  "answer": "...",
  "reasoning": "Briefly explain why the context is sufficient or insufficient. If insufficient, describe the missing or mismatched evidence."
}"""


class BaseAdapter(ABC):
    """Base class for all dataset adapters"""
    
    def __init__(self, raw_file_path: str):
        self.raw_file_path = raw_file_path
        self.logger = get_logger()
    
    def _format_context_blocks(self, context_blocks: List[str]) -> str:
        """
        Format context blocks with clear separators.
        
        Args:
            context_blocks: List of context text blocks
            
        Returns:
            Formatted context string with separators
        """
        if not context_blocks:
            return "No relevant context found."
        
        formatted_blocks = []
        for i, block in enumerate(context_blocks, 1):
            formatted_blocks.append(f"--- BEGIN CONTEXT BLOCK {i} ---")
            formatted_blocks.append(block)
            formatted_blocks.append(f"--- END CONTEXT BLOCK {i} ---")
        
        return "\n\n".join(formatted_blocks)

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
