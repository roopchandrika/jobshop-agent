"""Plant knowledge for the agent to look things up in (Phase 8: retrieval).

The solver knows the schedule; it does not know *why* the plant works the way it does. Procedures,
past incidents and policies live in documents. This package splits those documents into chunks,
indexes them, and answers "which passages are about this question?". It never imports an LLM SDK:
retrieval is plain, testable code, and a document is data, never an instruction.
"""

from jobshop.knowledge.base import Chunk, Hit, KnowledgeBase
from jobshop.knowledge.chunking import chunk_markdown
from jobshop.knowledge.loading import load_knowledge

__all__ = ["Chunk", "Hit", "KnowledgeBase", "chunk_markdown", "load_knowledge"]
