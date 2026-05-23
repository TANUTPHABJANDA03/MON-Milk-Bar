# rag_engine.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class RAGEngine:
    def __init__(self, kb_path: str):
        # โหลดโมเดลสำหรับแปลงข้อความเป็น Vector (Embedding)
        self.model = SentenceTransformer(EMBED_MODEL)
        # 1. Load & 2. Chunk: โหลดเอกสารและหั่นเป็นชิ้นๆ
        self.chunks = self._load_and_chunk(kb_path)
        # 3. Embed: แปลงร่างชิ้นส่วนข้อความทั้งหมดให้กลายเป็นดัชนีตัวเลข Vector
        self.index, self.embeddings = self._build_index()

    def _load_and_chunk(self, path: str) -> list[str]:
        """Load (ขั้นตอน 1) และ Chunk (ขั้นตอน 2)"""
        with open(path, encoding="utf-8") as f:
            text = f.read()
        # หั่นชิ้นข้อมูลโดยตัดตามบรรทัดเปล่า (\n\n)
        return [c.strip() for c in text.split("\n\n") if c.strip()]

    def _build_index(self):
        """Embed (ขั้นตอน 3)"""
        embeddings = self.model.encode(self.chunks, show_progress_bar=False)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype="float32"))
        return index, embeddings

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Search (ขั้นตอน 4)"""
        q_emb = self.model.encode([query])
        _, indices = self.index.search(np.array(q_emb, dtype="float32"), top_k)
        return [self.chunks[i] for i in indices[0]]