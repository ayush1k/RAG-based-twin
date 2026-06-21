"""
Lightweight retrieval over your personal knowledge base.

Uses TF-IDF + cosine similarity instead of a heavy embedding model.
This keeps the backend small and fast to deploy on a free-tier host
(no torch, no GPU, no extra API key). It works well at this scale
because the knowledge base is small and queries tend to share
keywords with the source text (project names, tech stack, company
names, etc).

If you outgrow this later, swap KnowledgeBase.search() for a real
embedding model (e.g. sentence-transformers or a hosted embeddings
API) without changing anything in main.py.
"""

import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"


class KnowledgeBase:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self._load(data_dir)

        if not self.chunks:
            raise RuntimeError(
                f"No knowledge base content found in {data_dir}. "
                "Add some .md files or a knowledge_base.json first."
            )

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def _load(self, data_dir: Path) -> None:
        """Load the knowledge base. Loads from knowledge_base.json if it exists,
        or auto-compiles from .md files if they are newer or if JSON is missing.
        """
        json_path = data_dir / "knowledge_base.json"
        md_files = sorted(data_dir.glob("*.md"))
        json_exists = json_path.exists()

        should_recompile = False
        if md_files:
            if json_exists:
                json_mtime = json_path.stat().st_mtime
                for md_path in md_files:
                    if md_path.stat().st_mtime > json_mtime:
                        should_recompile = True
                        break
            else:
                should_recompile = True
        elif not json_exists:
            raise RuntimeError(
                f"No knowledge base content found in {data_dir}. "
                "Add knowledge_base.json or some .md files."
            )

        if should_recompile and md_files:
            # Load from markdown, compile to json
            self.chunks = []
            self.sources = []
            for path in md_files:
                text = path.read_text(encoding="utf-8")
                for chunk in self._split(text):
                    chunk = chunk.strip()
                    if chunk:
                        self.chunks.append(chunk)
                        self.sources.append(path.stem)
            
            # Save compiled database to JSON
            kb_data = [{"text": chunk, "source": source} for chunk, source in zip(self.chunks, self.sources)]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(kb_data, f, indent=2, ensure_ascii=False)
        elif json_exists:
            # Load directly from JSON
            with open(json_path, "r", encoding="utf-8") as f:
                kb_data = json.load(f)
            self.chunks = [item["text"] for item in kb_data]
            self.sources = [item["source"] for item in kb_data]

    @staticmethod
    def _split(text: str) -> list[str]:
        # Split on blank lines so each chunk is roughly one paragraph
        # or one bullet/section — small enough to retrieve precisely,
        # big enough to keep context.
        return re.split(r"\n\s*\n", text)

    def search(self, query: str, k: int = 4, min_score: float = 0.05) -> list[str]:
        """Return up to k chunks most relevant to the query."""
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        ranked = sims.argsort()[::-1][:k]
        return [self.chunks[i] for i in ranked if sims[i] > min_score]

