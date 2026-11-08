import hashlib
import json
import math
import os
import random
import re
import shutil
import unicodedata
from collections import Counter, defaultdict

import faiss
import numpy as np
import ollama
import pymupdf

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
INDEX_DIR = "index"
RRF_K = 60  # constante de la fusion des classements (Reciprocal Rank Fusion)

SYSTEM_PROMPT = (
    "Tu es un assistant pédagogique. "
    "Réponds en français, de façon claire et simple, en utilisant "
    "UNIQUEMENT les informations du contexte fourni. "
    "N'ajoute aucune information extérieure et n'invente aucun exemple. "
    "Si le contexte ne contient pas la réponse, dis clairement que "
    "l'information est absente du document."
)


# 1. Moteurs d'embeddings
class MiniLMEmbedder:
    """Modèle local sentence-transformers (textes tronqués vers 500 caractères)."""
    name = "minilm"
    chunk_size = 500
    overlap = 100

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def encode(self, texts, is_query=False):
        vectors = self.model.encode(texts, convert_to_numpy=True)
        vectors = vectors.astype("float32")
        faiss.normalize_L2(vectors)
        return vectors


class OllamaEmbedder:
    """nomic-embed-text via Ollama : accepte des chunks plus longs."""
    name = "nomic"
    chunk_size = 800
    overlap = 150

    def __init__(self, model_name=OLLAMA_EMBEDDING_MODEL):
        self.model_name = model_name

    def encode(self, texts, is_query=False):
        # nomic-embed-text a été entraîné avec ces préfixes de tâche
        prefix = "search_query: " if is_query else "search_document: "
        texts = [prefix + t for t in texts]

        vectors = []
        for i in range(0, len(texts), 32):
            response = ollama.embed(model=self.model_name, input=texts[i:i + 32])
            vectors.extend(response["embeddings"])

        vectors = np.array(vectors, dtype="float32")
        faiss.normalize_L2(vectors)
        return vectors


EMBEDDERS = {
    "MiniLM (local)": "minilm",
    "nomic-embed-text (Ollama, plus précis)": "nomic",
}


def load_embedder(backend):
    if backend == "nomic":
        return OllamaEmbedder()
    return MiniLMEmbedder()


def index_dir_for(backend, base=INDEX_DIR):
    """Un dossier d'index par moteur d'embeddings."""
    return os.path.join(base, backend)


# 1b. Recherche par mots-clés (BM25)
STOPWORDS = set(
    "le la les un une des du de d l et ou en au aux ce ces cet cette qui que "
    "quoi dont est sont a ont fait faire il elle ils elles on je tu nous vous "
    "se sa son ses leur leurs sur sous dans par pour avec sans pas ne plus "
    "mais comme the of and to in is are for what how".split()
)


def tokenize(text):
    """Mots en minuscules, sans accents, sans mots vides."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return [w for w in re.findall(r"\w+", text) if w not in STOPWORDS]


class BM25:
    """Classement par mots-clés : retrouve les noms précis, symboles et code."""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(t)) for t in texts]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.n = len(texts)
        self.avgdl = (sum(self.lens) / self.n if self.n else 1.0) or 1.0

        document_frequency = Counter()
        self.postings = defaultdict(list)
        for i, tf in enumerate(self.tfs):
            document_frequency.update(tf.keys())
            for word, freq in tf.items():
                self.postings[word].append((i, freq))

        self.idf = {
            word: math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for word, df in document_frequency.items()
        }

    def scores(self, query):
        result = [0.0] * self.n

        for word in set(tokenize(query)):
            idf = self.idf.get(word)
            if idf is None:
                continue
            for i, freq in self.postings[word]:
                norm = freq + self.k1 * (
                    1 - self.b + self.b * self.lens[i] / self.avgdl
                )
                result[i] += idf * freq * (self.k1 + 1) / norm

        return result

    def coverage(self, query, i):
        """Part des mots de la question présents dans le chunk i."""
        words = set(tokenize(query))
        if not words:
            return 0.0
        return sum(1 for w in words if w in self.tfs[i]) / len(words)


# 2. Extraire le texte d'un PDF, page par page
def extract_pdf_pages(pdf_bytes):
    """Retourne une liste de tuples (numéro_de_page, texte)."""
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append((number, text))

    document.close()
    return pages


# 3. Découper un texte en morceaux sans couper les mots
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end < text_length:
            space_position = text.rfind(" ", start, end)
            if space_position > start:
                end = space_position

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(start + 1, end - overlap)

    return chunks


# 4. Construire tous les chunks (avec source et page) à partir des PDF
def build_chunks(documents, chunk_size=500, overlap=100):
    """documents : liste de tuples (nom_du_fichier, contenu_en_bytes)."""
    all_chunks = []

    for name, pdf_bytes in documents:
        for page_number, page_text in extract_pdf_pages(pdf_bytes):
            for chunk in chunk_text(page_text, chunk_size, overlap):
                all_chunks.append({
                    "text": chunk,
                    "source": name,
                    "page": page_number,
                })

    return all_chunks


# 5. Créer l'index FAISS
def create_faiss_index(chunks, embedder):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedder.encode(texts, is_query=False)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


# 5b. Sauvegarder / recharger l'index sur disque
def file_hash(data):
    """Empreinte du contenu d'un fichier (change si le PDF change)."""
    return hashlib.md5(data).hexdigest()


def save_index(index, chunks, signature, index_dir=INDEX_DIR):
    os.makedirs(index_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))

    with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump({"signature": signature, "chunks": chunks}, f, ensure_ascii=False)


def load_saved_index(index_dir=INDEX_DIR):
    """Retourne {"index", "chunks", "signature"} ou None si rien de valide."""
    index_path = os.path.join(index_dir, "index.faiss")
    chunks_path = os.path.join(index_dir, "chunks.json")

    if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
        return None

    try:
        index = faiss.read_index(index_path)
        with open(chunks_path, encoding="utf-8") as f:
            data = json.load(f)
        return {
            "index": index,
            "chunks": data["chunks"],
            "signature": data["signature"],
        }
    except Exception:
        return None


def delete_saved_index(index_dir=INDEX_DIR):
    shutil.rmtree(index_dir, ignore_errors=True)


# 6. Rechercher les passages pertinents (sens + mots-clés)
def search_chunks(question, chunks, embedder, index, k=4, min_score=0.3,
                  bm25=None, pool=30):
    """Recherche vectorielle, combinée à BM25 si `bm25` est fourni.

    Un passage est gardé si sa similarité de sens atteint `min_score`, ou
    s'il contient au moins la moitié des mots importants de la question.
    """
    n = len(chunks)
    if n == 0:
        return []

    query_vector = embedder.encode([question], is_query=True)
    scores, ids = index.search(query_vector, n)

    cosine = {}
    order = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            cosine[int(idx)] = float(score)
            order.append(int(idx))

    vector_rank = {idx: r for r, idx in enumerate(order[:pool])}

    keyword_rank = {}
    coverage = {}
    if bm25 is not None:
        keyword_scores = bm25.scores(question)
        ranked = sorted(range(n), key=lambda i: keyword_scores[i], reverse=True)

        for r, idx in enumerate(ranked[:pool]):
            if keyword_scores[idx] > 0:
                keyword_rank[idx] = r
                coverage[idx] = bm25.coverage(question, idx)

    # Fusion des deux classements (Reciprocal Rank Fusion)
    fused = {}
    for ranking in (vector_rank, keyword_rank):
        for idx, r in ranking.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + r)

    results = []

    for idx in sorted(fused, key=fused.get, reverse=True):
        by_meaning = cosine.get(idx, 0.0) >= min_score
        by_keywords = idx in keyword_rank and coverage[idx] >= 0.5

        if not (by_meaning or by_keywords):
            continue

        if by_meaning and by_keywords:
            via = "sens + mots-clés"
        elif by_meaning:
            via = "sens"
        else:
            via = "mots-clés"

        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": cosine.get(idx, 0.0),
            "via": via,
        })

        if len(results) >= k:
            break

    return results


# 7. Générer la réponse avec Ollama
def generate_answer(question, results, llm_model="llama3.2"):
    context = "\n\n".join(
        f"[{r['source']}, page {r['page']}]\n{r['text']}"
        for r in results
    )

    response = ollama.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Contexte du cours :\n{context}\n\n"
                    f"Question : {question}"
                ),
            },
        ],
        options={"temperature": 0.1},
    )

    return response["message"]["content"]


# ============================================================
# 8. Fonctions d'étude : quiz, flashcards, fiche de révision
# ============================================================

def select_study_chunks(chunks, index, model, source=None, topic="", n=8,
                        bm25=None):
    """Choisit les passages à donner au modèle.

    - Si un sujet est précisé : les passages les plus proches du sujet.
    - Sinon : n passages répartis régulièrement dans le document.
    """
    if topic.strip():
        found = search_chunks(
            topic, chunks, model, index, k=n * 4, min_score=0.0, bm25=bm25
        )
        if source:
            found = [r for r in found if r["source"] == source]
        return found[:n]

    pool = [c for c in chunks if source is None or c["source"] == source]

    if len(pool) <= n:
        return pool

    step = len(pool) / n
    return [pool[int(i * step)] for i in range(n)]


def _format_context(passages):
    return "\n\n".join(
        f"[{p['source']}, page {p['page']}]\n{p['text']}"
        for p in passages
    )


def _ask_llm(system, prompt, llm_model, temperature=0.3, json_mode=False):
    kwargs = {"format": "json"} if json_mode else {}

    response = ollama.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": temperature, "num_ctx": 4096},
        **kwargs,
    )

    return response["message"]["content"]


def _parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _normalize_answer(answer, options):
    """Retourne l'indice de la bonne réponse, ou None si illisible."""
    if isinstance(answer, bool):
        return None

    if isinstance(answer, int):
        return answer if 0 <= answer < len(options) else None

    if isinstance(answer, str):
        a = answer.strip()

        if a in options:
            return options.index(a)

        if a.isdigit() and int(a) < len(options):
            return int(a)

        if a and a[0].upper() in "ABCD" and (len(a) == 1 or a[1] in ").:-"):
            idx = "ABCD".index(a[0].upper())
            return idx if idx < len(options) else None

    return None


QUIZ_SYSTEM = (
    "Tu es un professeur qui crée des quiz à partir d'un cours. "
    "Tu réponds uniquement avec du JSON valide, sans texte autour."
)


def generate_quiz(passages, n_questions=5, llm_model="llama3.2"):
    """Retourne une liste de dicts : question, options, answer, explanation."""
    prompt = (
        f"Contexte du cours :\n{_format_context(passages)}\n\n"
        f"Crée {n_questions} questions à choix multiple en français, "
        "basées UNIQUEMENT sur ce contexte. Règles : exactement 4 "
        "propositions par question, une seule est correcte, et ne mets "
        "pas de lettres (A, B, C) devant les propositions. "
        "Réponds avec ce format JSON :\n"
        '{"questions": [{"question": "...", '
        '"options": ["...", "...", "...", "..."], '
        '"answer": 0, "explanation": "..."}]}\n'
        'Le champ "answer" est l\'indice (0 à 3) de la bonne proposition.'
    )

    for _ in range(2):
        try:
            data = _parse_json(
                _ask_llm(QUIZ_SYSTEM, prompt, llm_model, 0.3, json_mode=True)
            )
            raw = data.get("questions", []) if isinstance(data, dict) else data

            quiz = []
            for q in raw:
                question = str(q.get("question", "")).strip()
                options = [
                    str(o).strip() for o in q.get("options", [])
                    if str(o).strip()
                ]
                answer = _normalize_answer(q.get("answer"), options)

                if not question or len(options) < 2 or answer is None:
                    continue

                # Mélanger pour que la bonne réponse ne soit pas toujours la 1re
                correct = options[answer]
                random.shuffle(options)

                quiz.append({
                    "question": question,
                    "options": options,
                    "answer": options.index(correct),
                    "explanation": str(q.get("explanation", "")).strip(),
                })

            if quiz:
                return quiz[:n_questions]

        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            pass

    raise ValueError(
        "Le modèle n'a pas produit un quiz exploitable. "
        "Réessaie, ou utilise un modèle plus grand."
    )


def generate_flashcards(passages, n_cards=8, llm_model="llama3.2"):
    """Retourne une liste de dicts : question, answer."""
    prompt = (
        f"Contexte du cours :\n{_format_context(passages)}\n\n"
        f"Crée {n_cards} flashcards en français, basées UNIQUEMENT sur ce "
        "contexte : une question courte au recto, une réponse concise "
        "(1 à 2 phrases) au verso. Réponds avec ce format JSON :\n"
        '{"cards": [{"question": "...", "answer": "..."}]}'
    )

    system = (
        "Tu crées des flashcards de révision. "
        "Tu réponds uniquement avec du JSON valide, sans texte autour."
    )

    for _ in range(2):
        try:
            data = _parse_json(
                _ask_llm(system, prompt, llm_model, 0.3, json_mode=True)
            )
            raw = data.get("cards", []) if isinstance(data, dict) else data

            cards = []
            for c in raw:
                question = str(c.get("question", "")).strip()
                answer = str(c.get("answer", "")).strip()
                if question and answer:
                    cards.append({"question": question, "answer": answer})

            if cards:
                return cards[:n_cards]

        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            pass

    raise ValueError(
        "Le modèle n'a pas produit de flashcards exploitables. "
        "Réessaie, ou utilise un modèle plus grand."
    )


def generate_summary(passages, llm_model="llama3.2"):
    """Retourne une fiche de révision en Markdown."""
    system = (
        "Tu es un assistant pédagogique. Tu rédiges en français, "
        "uniquement à partir du contexte fourni, sans rien inventer."
    )

    prompt = (
        f"Contexte du cours :\n{_format_context(passages)}\n\n"
        "Rédige une fiche de révision structurée en Markdown : un titre, "
        "des sous-titres (##), les points clés en puces, et les définitions "
        "importantes en gras. Reste fidèle au contexte et tiens sur "
        "environ une page."
    )

    return _ask_llm(system, prompt, llm_model, temperature=0.2)