"""
summarizer.py
-------------
Stage: Smart Chunking -> Transformer Summarizer -> Final Summary

Hierarchical (map-reduce) abstractive summarisation:
  1. summarise every chunk,
  2. concatenate the chunk summaries,
  3. if that is still too long, re-chunk and repeat; otherwise one final pass.

Uses AutoModelForSeq2SeqLM.generate() directly (works across transformers
versions, unlike the 'summarization' pipeline task).
"""
import re

MODELS = {
    "fast": "sshleifer/distilbart-cnn-12-6",   # ~300MB, good default on CPU
    "best": "facebook/bart-large-cnn",         # ~1.6GB, higher quality
}


class Summarizer:
    def __init__(self, model_name: str = MODELS["fast"], max_input_tokens: int = 900):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()
        self.max_input_tokens = max_input_tokens

    def count_tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def _generate(self, text: str, max_len: int, min_len: int) -> str:
        inputs = self.tok(text, return_tensors="pt", truncation=True,
                          max_length=self.max_input_tokens).to(self.device)
        with self.torch.no_grad():
            ids = self.model.generate(
                **inputs, num_beams=4, max_length=max_len, min_length=min_len,
                length_penalty=2.0, no_repeat_ngram_size=3, early_stopping=True,
            )
        out = self.tok.decode(ids[0], skip_special_tokens=True).strip()
        return re.sub(r"\s+", " ", out)

    def summarize_text(self, text: str, max_len: int = 160) -> str:
        n = self.count_tokens(text)
        if n < 60:                       # already short: don't "summarise" it
            return text
        max_len = int(min(max_len, max(40, n * 0.4)))
        min_len = max(20, int(max_len * 0.4))
        return self._generate(text, max_len, min_len)

    def summarize_chunks(self, chunks: list[dict]) -> list[dict]:
        for c in chunks:
            c["summary"] = self.summarize_text(c["text"], max_len=150)
        return chunks

    def final_summary(self, chunk_summaries: list[str], final_max_len: int = 220,
                      _depth: int = 0) -> str:
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]
        joined = " ".join(chunk_summaries)
        if self.count_tokens(joined) <= self.max_input_tokens or _depth >= 3:
            return self.summarize_text(joined, max_len=final_max_len)
        # still too long -> summarise groups of summaries, then recurse
        groups, cur, cur_n = [], [], 0
        for s in chunk_summaries:
            n = self.count_tokens(s)
            if cur and cur_n + n > self.max_input_tokens - 50:
                groups.append(" ".join(cur)); cur, cur_n = [], 0
            cur.append(s); cur_n += n
        if cur:
            groups.append(" ".join(cur))
        return self.final_summary([self.summarize_text(g, 150) for g in groups],
                                  final_max_len, _depth + 1)
