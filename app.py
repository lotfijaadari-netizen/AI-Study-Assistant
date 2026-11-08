import csv
import glob
import io
import os

import streamlit as st

import rag

# --------------------------------
# Configuration de la page
# --------------------------------

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 AI Study Assistant")
st.write("Pose des questions sur tes cours PDF et révise-les.")


@st.cache_resource(show_spinner="Chargement du modèle d'embeddings...")
def get_embedder(backend):
    return rag.load_embedder(backend)


# --------------------------------
# 1. Barre latérale : mode, documents et réglages
# --------------------------------

MODES = ["💬 Chat", "📝 Quiz", "🗂️ Flashcards", "📄 Fiche de révision"]
mode = st.sidebar.radio("Mode", MODES)

st.sidebar.header("Mes documents")

uploaded_files = st.sidebar.file_uploader(
    "Importer des cours PDF",
    type=["pdf"],
    accept_multiple_files=True
)

use_folder = st.sidebar.checkbox("Inclure le dossier documents/")

st.sidebar.header("Recherche")

embedder_label = st.sidebar.selectbox(
    "Moteur d'embeddings", list(rag.EMBEDDERS)
)
backend = rag.EMBEDDERS[embedder_label]

hybrid = st.sidebar.checkbox(
    "Recherche hybride (sens + mots-clés)", value=True
)

st.sidebar.header("Réglages")

llm_model = st.sidebar.text_input("Modèle Ollama", value="llama3.2")

top_k = st.sidebar.slider(
    "Chat : nombre de passages utilisés", min_value=1, max_value=8, value=4
)

# nomic donne des similarités plus élevées : seuil par défaut plus haut
min_score = st.sidebar.slider(
    "Chat : score minimum de pertinence",
    min_value=0.0, max_value=1.0,
    value=0.5 if backend == "nomic" else 0.3,
    step=0.05,
    key=f"min_score_{backend}"
)

# Rassembler tous les documents (nom, contenu)
documents = [(f.name, f.getvalue()) for f in uploaded_files]

if use_folder and os.path.isdir("documents"):
    for path in sorted(glob.glob(os.path.join("documents", "*.pdf"))):
        with open(path, "rb") as f:
            documents.append((os.path.basename(path), f.read()))

# --------------------------------
# 2. Construire (ou recharger) l'index
# --------------------------------

INDEX_KEYS = ("signature", "chunks", "index", "bm25", "bm25_sig")


def clear_index():
    for key in INDEX_KEYS:
        st.session_state.pop(key, None)


if st.sidebar.button("Effacer les index sauvegardés"):
    rag.delete_saved_index()
    clear_index()
    st.rerun()

# Empreinte : change si un PDF change ou si le moteur d'embeddings change
signature = {
    "embedder": backend,
    "docs": sorted([name, rag.file_hash(data)] for name, data in documents),
}
index_dir = rag.index_dir_for(backend)

if not documents:
    clear_index()

elif st.session_state.get("signature") != signature:
    try:
        saved = rag.load_saved_index(index_dir)

        if saved is not None and saved["signature"] == signature:
            # Mêmes PDF et même moteur : on recharge depuis le disque
            st.session_state.index = saved["index"]
            st.session_state.chunks = saved["chunks"]
            st.session_state.signature = signature
            st.sidebar.caption("Index rechargé depuis le disque.")
        else:
            with st.spinner("Lecture et indexation des cours..."):
                embedder = get_embedder(backend)
                chunks = rag.build_chunks(
                    documents, embedder.chunk_size, embedder.overlap
                )

                if chunks:
                    index = rag.create_faiss_index(chunks, embedder)
                    rag.save_index(index, chunks, signature, index_dir)
                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.signature = signature
                    st.sidebar.caption("Index créé et sauvegardé.")
                else:
                    clear_index()
                    st.warning(
                        "Aucun texte extrait. "
                        "Ces PDF sont peut-être des documents scannés."
                    )
    except Exception as e:
        st.error(
            f"Erreur lors de l'indexation : {e}\n\n"
            "Pour nomic-embed-text, vérifie qu'Ollama tourne et que le "
            "modèle est installé : `ollama pull nomic-embed-text`."
        )

# Index des mots-clés (BM25) : reconstruit quand l'index change
if (
    "chunks" in st.session_state
    and st.session_state.get("bm25_sig") != st.session_state.get("signature")
):
    st.session_state.bm25 = rag.BM25(
        [c["text"] for c in st.session_state.chunks]
    )
    st.session_state.bm25_sig = st.session_state.get("signature")


def current_bm25():
    """BM25 si la recherche hybride est activée, sinon None."""
    return st.session_state.get("bm25") if hybrid else None


index_ready = "index" in st.session_state

if index_ready:
    nb_docs = len({c["source"] for c in st.session_state.chunks})
    st.sidebar.success(
        f"{nb_docs} document(s) indexé(s), "
        f"{len(st.session_state.chunks)} morceaux"
    )
else:
    st.info("Importe un PDF dans la barre latérale pour commencer.")


# --------------------------------
# 3. Mode Chat
# --------------------------------

def show_sources(sources):
    if not sources:
        return
    with st.expander("Passages utilisés"):
        for s in sources:
            via = f", trouvé par : {s['via']}" if s.get("via") else ""
            st.markdown(
                f"**{s['source']}**, page {s['page']} "
                f"(score : {s['score']:.2f}{via})"
            )
            st.text(s["text"])


def chat_page():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            show_sources(message.get("sources"))

    question = st.chat_input("Pose ta question sur le cours...")

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if not index_ready:
            answer = "Importe d'abord un PDF dans la barre latérale."
            sources = []
            st.write(answer)
        else:
            try:
                model = get_embedder(backend)

                with st.spinner("Recherche dans le cours..."):
                    results = rag.search_chunks(
                        question,
                        st.session_state.chunks,
                        model,
                        st.session_state.index,
                        k=top_k,
                        min_score=min_score,
                        bm25=current_bm25()
                    )

                if not results:
                    best = rag.search_chunks(
                        question,
                        st.session_state.chunks,
                        model,
                        st.session_state.index,
                        k=1,
                        min_score=0.0,
                        bm25=None
                    )
                    best_score = best[0]["score"] if best else 0.0
                    answer = (
                        "Je n'ai trouvé aucun passage pertinent "
                        f"(meilleur score : {best_score:.2f}, "
                        f"seuil : {min_score:.2f}). "
                        "Essaie de reformuler ta question ou de baisser "
                        "le score minimum."
                    )
                    sources = []
                else:
                    with st.spinner("Llama rédige la réponse..."):
                        answer = rag.generate_answer(
                            question, results, llm_model=llm_model
                        )
                    sources = results

                st.write(answer)
                show_sources(sources)

            except Exception as e:
                answer = (
                    f"Erreur : {e}\n\n"
                    "Vérifie qu'Ollama tourne bien et que le modèle "
                    f"« {llm_model} » est installé."
                )
                sources = []
                st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })


# --------------------------------
# 4. Modes d'étude : quiz, flashcards, fiche
# --------------------------------

def study_controls():
    """Choix du document, du sujet et du nombre de passages."""
    sources = sorted({c["source"] for c in st.session_state.chunks})
    options = ["Tous les documents"] + sources

    choice = st.selectbox("Document", options)
    source = None if choice == options[0] else choice

    topic = st.text_input(
        "Sujet précis (optionnel)",
        placeholder="ex : codage Manchester"
    )

    n_chunks = st.slider("Passages du cours utilisés", 4, 12, 8)

    return source, topic, n_chunks


def get_passages(source, topic, n_chunks):
    passages = rag.select_study_chunks(
        st.session_state.chunks,
        st.session_state.index,
        get_embedder(backend),
        source=source,
        topic=topic,
        n=n_chunks,
        bm25=current_bm25()
    )
    if not passages:
        st.warning("Aucun passage trouvé pour ce document ou ce sujet.")
    return passages


def llm_error(e):
    st.error(
        f"Erreur : {e}\n\n"
        "Vérifie qu'Ollama tourne bien et que le modèle "
        f"« {llm_model} » est installé."
    )


def quiz_page():
    st.subheader("📝 Quiz")
    source, topic, n_chunks = study_controls()
    n_questions = st.slider("Nombre de questions", 3, 10, 5)

    if st.button("Générer le quiz", type="primary"):
        try:
            with st.spinner("Llama prépare le quiz (1 à 2 minutes)..."):
                passages = get_passages(source, topic, n_chunks)
                quiz = (
                    rag.generate_quiz(passages, n_questions, llm_model)
                    if passages else None
                )

            if quiz:
                for key in [k for k in st.session_state if k.startswith("qz_")]:
                    del st.session_state[key]
                st.session_state.quiz = quiz
                st.session_state.quiz_checked = False
        except Exception as e:
            llm_error(e)

    quiz = st.session_state.get("quiz")
    if not quiz:
        return

    st.divider()

    for i, q in enumerate(quiz):
        st.markdown(f"**{i + 1}. {q['question']}**")
        st.radio(
            "Réponse",
            q["options"],
            index=None,
            key=f"qz_{i}",
            label_visibility="collapsed"
        )

    if st.button("Corriger"):
        st.session_state.quiz_checked = True

    if st.session_state.get("quiz_checked"):
        score = 0

        for i, q in enumerate(quiz):
            choice = st.session_state.get(f"qz_{i}")
            correct = q["options"][q["answer"]]

            if choice == correct:
                score += 1
                st.success(f"Question {i + 1} : correct ✅")
            else:
                st.error(
                    f"Question {i + 1} : la bonne réponse était « {correct} »"
                )

            if q.get("explanation"):
                st.caption(q["explanation"])

        st.subheader(f"Score : {score} / {len(quiz)}")


def flashcards_page():
    st.subheader("🗂️ Flashcards")
    source, topic, n_chunks = study_controls()
    n_cards = st.slider("Nombre de flashcards", 4, 15, 8)

    if st.button("Générer les flashcards", type="primary"):
        try:
            with st.spinner("Llama crée les flashcards (1 à 2 minutes)..."):
                passages = get_passages(source, topic, n_chunks)
                cards = (
                    rag.generate_flashcards(passages, n_cards, llm_model)
                    if passages else None
                )

            if cards:
                st.session_state.cards = cards
        except Exception as e:
            llm_error(e)

    cards = st.session_state.get("cards")
    if not cards:
        return

    st.divider()
    st.caption("Clique sur une question pour voir la réponse.")

    for i, card in enumerate(cards, start=1):
        with st.expander(f"{i}. {card['question']}"):
            st.write(card["answer"])

    # Export compatible avec l'import d'Anki (séparateur tabulation)
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t")
    for card in cards:
        writer.writerow([card["question"], card["answer"]])

    st.download_button(
        "Télécharger pour Anki (.tsv)",
        data=buffer.getvalue().encode("utf-8"),
        file_name="flashcards.tsv",
        mime="text/tab-separated-values"
    )


def summary_page():
    st.subheader("📄 Fiche de révision")
    source, topic, n_chunks = study_controls()

    if st.button("Générer la fiche", type="primary"):
        try:
            with st.spinner("Llama rédige la fiche (1 à 2 minutes)..."):
                passages = get_passages(source, topic, n_chunks)
                summary = (
                    rag.generate_summary(passages, llm_model)
                    if passages else None
                )

            if summary:
                st.session_state.summary = summary
        except Exception as e:
            llm_error(e)

    summary = st.session_state.get("summary")
    if not summary:
        return

    st.divider()
    st.markdown(summary)

    st.download_button(
        "Télécharger la fiche (.md)",
        data=summary.encode("utf-8"),
        file_name="fiche_de_revision.md",
        mime="text/markdown"
    )


# --------------------------------
# 5. Afficher le mode choisi
# --------------------------------

if mode == MODES[0]:
    chat_page()
elif index_ready:
    if mode == MODES[1]:
        quiz_page()
    elif mode == MODES[2]:
        flashcards_page()
    else:
        summary_page()