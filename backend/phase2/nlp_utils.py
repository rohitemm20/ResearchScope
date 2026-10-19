"""
nlp_utils.py
------------
Stage: Text Cleaning -> spaCy Tokenization

Splits cleaned paragraphs into sentences with spaCy, repairs the common
abbreviation mis-splits ("et al.", "Fig."), and attaches content tokens.
"""
import re
from functools import lru_cache

ABBREV_END = re.compile(
    r"(?:\bet al|\be\.g|\bi\.e|\bFigs?|\bEqs?|\bvs|\betc|\bapprox|\bRs|\bNo)\.$",
    re.I,
)


@lru_cache(maxsize=1)
def get_nlp():
    """en_core_web_sm if installed, otherwise a rule-based fallback."""
    import spacy
    try:
        return spacy.load("en_core_web_sm", disable=["ner"])
    except OSError:
        print("[nlp_utils] en_core_web_sm not found -> using blank model + "
              "sentencizer. Run: python -m spacy download en_core_web_sm")
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp


def content_tokens(doc_or_span) -> list[str]:
    """Lower-cased alphabetic, non-stopword tokens."""
    return [t.text.lower() for t in doc_or_span
            if t.is_alpha and not t.is_stop and len(t.text) > 1]


def split_sentences(paragraphs: list[dict], min_words: int = 6,
                    min_alpha_ratio: float = 0.55) -> list[dict]:
    """
    paragraphs: output of text_cleaning.clean_document()
    returns   : [{"id","text","page","section","para_idx","tokens"}]
    """
    nlp = get_nlp()
    sentences: list[dict] = []

    def emit(text, para, pi, span):
        words = text.split()
        letters = sum(c.isalpha() for c in text)
        if len(words) < min_words or letters / max(len(text), 1) < min_alpha_ratio:
            return
        sentences.append({
            "id": len(sentences),
            "text": text,
            "page": para["page"],
            "section": para["section"],
            "para_idx": pi,
            "tokens": content_tokens(span),
        })

    for pi, para in enumerate(paragraphs):
        doc = nlp(para["text"])
        buf, start = "", None
        for sent in doc.sents:
            t = sent.text.strip()
            if not t:
                continue
            if not buf:
                start = sent.start
            buf = f"{buf} {t}".strip()
            if ABBREV_END.search(buf):       # "Zhong et al." -> keep going
                continue
            emit(buf, para, pi, doc[start:sent.end])
            buf = ""
        if buf:
            emit(buf, para, pi, doc[start:])
    return sentences
