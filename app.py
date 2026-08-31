import os
import tempfile
import uuid
from datetime import datetime

import streamlit as st

from src.document_loader import load_document
from src.text_processor import clean_text, split_pages
from src.embeddings import generate_embeddings
from src.vector_store import create_faiss_index

from src.rag_pipeline import (
    retrieve_context,
    build_prompt,
    generate_answer,
    summarize_document,
    build_source_citations
)

from src.sentiment import analyze_sentiment
from src.intent import analyze_intent


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IntelliAssist AI",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

DEFAULT_SETTINGS = {
    "model": "gemini-3.6-flash",
    "temperature": 0.0,
    "top_k": 3,
    "distance_threshold": 1.40,
    "chat_history_enabled": True,
    "sentiment_enabled": True,
    "intent_enabled": True,
    "source_citations_enabled": True,
    "semantic_search_enabled": True
}


def initialize_state():

    defaults = {

        "document_ready": False,

        "chunks": [],

        "index": None,

        "page_count": 0,

        "document_name": "",

        "summary": None,

        # Current conversation
        "messages": [],

        # Saved conversations
        "chat_history": [],

        # Currently selected conversation
        "current_chat_id": None,

        "last_question": "",

        "last_context": [],

        "last_sentiment": None,

        "last_intent": None,

        "settings": DEFAULT_SETTINGS.copy(),

        "active_page": "Chat"
    }


    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


initialize_state()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_chat_id():
    return str(uuid.uuid4())


def create_chat_title(question):

    if not question:
        return "New Conversation"

    title = question.strip()

    if len(title) > 45:
        title = title[:45].rstrip() + "..."

    return title


def save_current_conversation():

    if not st.session_state["settings"]["chat_history_enabled"]:
        return

    messages = st.session_state["messages"]

    if not messages:
        return


    chat_id = st.session_state.get(
        "current_chat_id"
    )


    if not chat_id:

        chat_id = create_chat_id()

        st.session_state["current_chat_id"] = chat_id


    title = "New Conversation"

    for message in messages:

        if message.get("role") == "user":

            title = create_chat_title(
                message.get("content", "")
            )

            break


    existing_index = None

    for i, conversation in enumerate(
        st.session_state["chat_history"]
    ):

        if conversation.get("id") == chat_id:

            existing_index = i

            break


    conversation_data = {

        "id": chat_id,

        "title": title,

        "created_at": (
            datetime.now().strftime(
                "%d %b %Y, %I:%M %p"
            )
        ),

        "messages": [
            {
                "role": message.get("role"),
                "content": message.get("content", "")
            }
            for message in messages
        ]
    }


    if existing_index is not None:

        st.session_state[
            "chat_history"
        ][existing_index] = conversation_data

    else:

        st.session_state[
            "chat_history"
        ].insert(
            0,
            conversation_data
        )


def start_new_chat():

    # Save current conversation before starting another.
    save_current_conversation()

    st.session_state["messages"] = []

    st.session_state["current_chat_id"] = None

    st.session_state["last_question"] = ""

    st.session_state["last_context"] = []

    st.session_state["last_sentiment"] = None

    st.session_state["last_intent"] = ""


def load_conversation(chat_id):

    for conversation in st.session_state["chat_history"]:

        if conversation.get("id") == chat_id:

            st.session_state["messages"] = [
                {
                    "role": message.get("role"),
                    "content": message.get("content", "")
                }
                for message in conversation.get(
                    "messages",
                    []
                )
            ]

            st.session_state[
                "current_chat_id"
            ] = chat_id

            st.session_state[
                "last_question"
            ] = ""

            st.session_state[
                "last_context"
            ] = []

            st.session_state[
                "last_sentiment"
            ] = None

            st.session_state[
                "last_intent"
            ] = None

            return


def clear_current_chat():

    st.session_state["messages"] = []

    st.session_state["current_chat_id"] = None

    st.session_state["last_question"] = ""

    st.session_state["last_context"] = []

    st.session_state["last_sentiment"] = None

    st.session_state["last_intent"] = None


def clear_all_history():

    st.session_state["chat_history"] = []

    clear_current_chat()


def quota_message():

    return (
        "⚠️ **AI service quota is currently exhausted.**\n\n"
        "Please try again later or check your Gemini "
        "API quota and billing settings."
    )

# ============================================================
# ERROR MESSAGES
# ============================================================

NO_DOCUMENT_MESSAGE = (
    "⚠ Please upload a document first."
)


EMPTY_DOCUMENT_MESSAGE = (
    "⚠ No readable text found."
)


QUESTION_NOT_FOUND_MESSAGE = (
    "I couldn't find this information in\n"
    "the uploaded document."
)


API_FAILURE_MESSAGE = (
    "⚠ AI service temporarily unavailable.\n"
    "Please try again."
)


def is_api_failure(response):

    if not isinstance(response, str):
        return False

    return (
        "⚠ AI service temporarily unavailable."
        in response
    )


# ============================================================
# HEADER
# ============================================================

st.title("🤖 IntelliAssist AI")

st.caption(
    "Your Intelligent Document Assistant"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤖 IntelliAssist AI")

    st.divider()


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    st.write("### 💬 Chat History")


    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        start_new_chat()

        st.rerun()


    if not st.session_state["chat_history"]:

        st.caption(
            "No previous conversations."
        )

    else:

        for conversation in st.session_state[
            "chat_history"
        ]:

            chat_id = conversation.get(
                "id"
            )

            title = conversation.get(
                "title",
                "Conversation"
            )


            is_active = (
                chat_id
                ==
                st.session_state.get(
                    "current_chat_id"
                )
            )


            button_label = (
                "🟢 "
                if is_active
                else "💬 "
            ) + title


            if st.button(
                button_label,
                key=f"history_{chat_id}",
                use_container_width=True
            ):

                load_conversation(
                    chat_id
                )

                st.rerun()


    st.divider()


    if st.button(
        "🗑 Clear History",
        use_container_width=True
    ):

        clear_all_history()

        st.success(
            "Chat history cleared."
        )

        st.rerun()


    st.divider()


    # ========================================================
    # CURRENT DOCUMENT
    # ========================================================

    st.write("### 📄 Current Document")


    if st.session_state["document_ready"]:

        st.success(
            st.session_state["document_name"]
        )

        st.caption(
            f"Pages: {st.session_state['page_count']}"
        )

        st.caption(
            f"Chunks: {len(st.session_state['chunks'])}"
        )

    else:

        st.info(
            "No document uploaded."
        )


    st.divider()


    # ========================================================
    # CURRENT CHAT
    # ========================================================

    st.write("### 💬 Current Chat")

    st.caption(
        f"Messages: "
        f"{len(st.session_state['messages'])}"
    )


    if st.button(
        "🗑 Clear Current Chat",
        use_container_width=True
    ):

        clear_current_chat()

        st.rerun()


# ============================================================
# DOCUMENT UPLOADER
# ============================================================

st.write("### 📁 Upload Document")


uploaded_file = st.file_uploader(
    "Choose PDF, TXT or DOCX",
    type=["pdf", "txt", "docx"],
    help="Supported formats: PDF, TXT, DOCX"
)


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

if uploaded_file is not None:

    previous_document = st.session_state.get(
        "document_name",
        ""
    )


    # ========================================================
    # NEW DOCUMENT DETECTION
    # ========================================================

    if previous_document != uploaded_file.name:

        st.session_state["document_ready"] = False

        st.session_state["chunks"] = []

        st.session_state["index"] = None

        st.session_state["page_count"] = 0

        st.session_state["summary"] = None

        # Do not destroy saved chat history.
        clear_current_chat()


    # ========================================================
    # PROCESS DOCUMENT
    # ========================================================

    if not st.session_state["document_ready"]:

        file_extension = os.path.splitext(
            uploaded_file.name
        )[1]

        temp_file_path = None


        try:

            # ==================================================
            # SAVE TEMP FILE
            # ==================================================

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_file_path = temp_file.name


            # ==================================================
            # PROCESS DOCUMENT
            # ==================================================

            with st.spinner(
                "📚 Processing document..."
            ):

                pages = load_document(
                    temp_file_path
                )


                if not pages:
                
                    st.warning(
                        EMPTY_DOCUMENT_MESSAGE
                    )

                    st.session_state["document_ready"] = False

                    st.session_state["chunks"] = []

                    st.session_state["index"] = None

                    st.session_state["page_count"] = 0

                    st.session_state["summary"] = None

                    st.stop()


                # ------------------------------------------------
                # NORMALIZE TXT / DOCX
                # ------------------------------------------------

                if isinstance(pages, str):

                    pages = [
                        {
                            "page": 1,
                            "text": pages,
                            "source": uploaded_file.name
                        }
                    ]


                # ------------------------------------------------
                # CLEAN PAGES
                # ------------------------------------------------

                cleaned_pages = []


                for page in pages:

                    text = clean_text(
                        page.get(
                            "text",
                            ""
                        )
                    )


                    if text.strip():

                        cleaned_pages.append(
                            {
                                "page": page.get(
                                    "page",
                                    1
                                ),

                                "source": uploaded_file.name,

                                "text": text,

                                "section": page.get(
                                    "section"
                                )
                            }
                        )


                if not cleaned_pages:
                
                    st.warning(
                        EMPTY_DOCUMENT_MESSAGE
                    )

                    st.session_state["document_ready"] = False

                    st.session_state["chunks"] = []

                    st.session_state["index"] = None

                    st.session_state["page_count"] = 0

                    st.session_state["summary"] = None

                    st.stop()


                # ------------------------------------------------
                # PAGE COUNT
                # ------------------------------------------------

                page_count = len(
                    cleaned_pages
                )


                # ------------------------------------------------
                # SPLIT CHUNKS
                # ------------------------------------------------

                chunks = split_pages(
                    cleaned_pages
                )


                if not chunks:

                    raise ValueError(
                        "The document could not be divided "
                        "into readable text chunks."
                    )


                # ------------------------------------------------
                # ENSURE METADATA
                # ------------------------------------------------

                for chunk in chunks:

                    if not chunk.get("source"):

                        chunk["source"] = (
                            uploaded_file.name
                        )


                    if not chunk.get("page"):

                        chunk["page"] = 1


                # ------------------------------------------------
                # CHUNK TEXT
                # ------------------------------------------------

                chunk_texts = [
                    chunk["text"]
                    for chunk in chunks
                    if chunk.get("text")
                ]


                if not chunk_texts:

                    raise ValueError(
                        "No readable text was found to "
                        "generate embeddings."
                    )


                # ------------------------------------------------
                # EMBEDDINGS
                # ------------------------------------------------

                embeddings = generate_embeddings(
                    chunk_texts
                )


                if embeddings is None:

                    raise ValueError(
                        "Unable to generate embeddings "
                        "for this document."
                    )


                if len(embeddings) == 0:

                    raise ValueError(
                        "Unable to generate embeddings."
                    )


                # ------------------------------------------------
                # FAISS
                # ------------------------------------------------

                index = create_faiss_index(
                    embeddings
                )


                # ------------------------------------------------
                # SAVE STATE
                # ------------------------------------------------

                st.session_state["chunks"] = chunks

                st.session_state["index"] = index

                st.session_state["page_count"] = page_count

                st.session_state["document_ready"] = True

                st.session_state["document_name"] = (
                    uploaded_file.name
                )

                st.session_state["summary"] = None


        except ValueError as e:

            st.error(
                f"❌ {str(e)}"
            )


            if file_extension.lower() == ".pdf":

                st.info(
                    "💡 This PDF may be scanned or may contain "
                    "images instead of selectable text."
                )

                st.warning(
                    "📄 Please upload a text-based PDF, "
                    "TXT, or DOCX file."
                )

            else:

                st.info(
                    "📄 Supported formats: PDF, TXT, DOCX"
                )


        except Exception as e:

            st.error(
                "❌ Unable to process this document."
            )

            st.caption(
                f"Technical details: {e}"
            )


        finally:

            try:

                if (
                    temp_file_path
                    and os.path.exists(temp_file_path)
                ):

                    os.remove(
                        temp_file_path
                    )

            except Exception:

                pass


# ============================================================
# STOP IF NO DOCUMENT
# ============================================================

if not st.session_state["document_ready"]:

    st.warning(
        NO_DOCUMENT_MESSAGE
    )

    st.stop()


# ============================================================
# DOCUMENT STATUS
# ============================================================

st.success(
    f"✅ {st.session_state['document_name']} is ready!"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "📄 Pages",
        st.session_state["page_count"]
    )


with col2:

    st.metric(
        "🧩 Chunks",
        len(st.session_state["chunks"])
    )


with col3:

    st.metric(
        "🔢 FAISS Vectors",
        st.session_state["index"].ntotal
    )


st.divider()


# ============================================================
# DASHBOARD TABS
# ============================================================

tab_chat, tab_documents, tab_summary, tab_search, tab_analysis, tab_settings = st.tabs(
    [
        "💬 Chat",
        "📄 Documents",
        "📝 Summary",
        "🔎 Search",
        "📊 Analysis",
        "⚙ Settings"
    ]
)


# ============================================================
# TAB 1 — CHAT
# ============================================================

with tab_chat:

    st.subheader(
        "💬 Chat with your document"
    )

    st.caption(
        "Ask questions and get answers based only "
        "on your uploaded document."
    )


    # ========================================================
    # CURRENT CONVERSATION
    # ========================================================

    if not st.session_state["messages"]:

        st.info(
            "👋 No conversation yet. "
            "Ask your first question below."
        )

    else:

        for message in st.session_state["messages"]:

            role = message.get(
                "role",
                ""
            )

            content = message.get(
                "content",
                ""
            )


            if role == "user":

                with st.chat_message("user"):

                    st.markdown(
                        content
                    )


            elif role == "assistant":

                with st.chat_message("assistant"):

                    st.markdown(
                        content
                    )


    # ========================================================
    # CHAT INPUT
    # ========================================================

    question = st.chat_input(
        "Ask something about your document..."
    )


    # ========================================================
    # PROCESS QUESTION
    # ========================================================

    if question:

        question = question.strip()


        if not question:

            st.stop()


        # ====================================================
        # CREATE CURRENT CONVERSATION
        # ====================================================

        if not st.session_state["current_chat_id"]:

            st.session_state[
                "current_chat_id"
            ] = create_chat_id()


        # ====================================================
        # SAVE USER MESSAGE
        # ====================================================

        st.session_state[
            "messages"
        ].append(
            {
                "role": "user",
                "content": question
            }
        )


        st.session_state[
            "last_question"
        ] = question


        # ====================================================
        # SENTIMENT ANALYSIS
        # ====================================================

        if st.session_state[
            "settings"
        ]["sentiment_enabled"]:

            try:

                sentiment_result = analyze_sentiment(
                    question
                )

                st.session_state[
                    "last_sentiment"
                ] = sentiment_result

            except Exception:

                st.session_state[
                    "last_sentiment"
                ] = {
                    "sentiment": "Neutral",
                    "confidence": 0.0
                }

        else:

            st.session_state[
                "last_sentiment"
            ] = None


        # ====================================================
        # INTENT ANALYSIS
        # ====================================================

        if st.session_state[
            "settings"
        ]["intent_enabled"]:

            try:

                intent_result = analyze_intent(
                    question
                )

                st.session_state[
                    "last_intent"
                ] = intent_result

            except Exception:

                st.session_state[
                    "last_intent"
                ] = {
                    "intent": "UNKNOWN",
                    "confidence": 0.0
                }

        else:

            st.session_state[
                "last_intent"
            ] = None


        # ====================================================
        # RETRIEVE DOCUMENT CONTEXT
        # ====================================================

        with st.spinner(
            "🔎 Searching the document..."
        ):

            try:

                context = retrieve_context(
                    question,
                    st.session_state["chunks"],
                    st.session_state["index"],
                    top_k=st.session_state[
                        "settings"
                    ]["top_k"],
                    distance_threshold=st.session_state[
                        "settings"
                    ]["distance_threshold"]
                )


                st.session_state[
                    "last_context"
                ] = context


            except Exception:

                error_message = API_FAILURE_MESSAGE


                st.session_state[
                    "messages"
                ].append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )


                save_current_conversation()


                st.warning(
                    error_message
                )


                st.stop()


        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        if not context:

            answer = QUESTION_NOT_FOUND_MESSAGE

        else:

            prompt = build_prompt(
                question,
                context
            )


            with st.spinner(
                "🤖 Generating AI answer..."
            ):

                try:

                    answer = generate_answer(
                        prompt,
                        temperature=st.session_state[
                            "settings"
                        ]["temperature"],
                        model=st.session_state[
                            "settings"
                        ]["model"]
                    )


                except Exception:

                    answer = API_FAILURE_MESSAGE


        # ====================================================
        # PHASE 24 — API FAILURE CHECK
        # ====================================================

        if is_api_failure(answer):

            answer = API_FAILURE_MESSAGE



        # ====================================================
        # SOURCE CITATIONS
        # ====================================================

        if (
            context
            and
            not is_api_failure(answer)
            and
            answer != QUESTION_NOT_FOUND_MESSAGE
            and
            st.session_state[
                "settings"
            ]["source_citations_enabled"]
        ):

            try:

                citations = build_source_citations(
                    context
                )

            except Exception:

                citations = []


            if citations:

                answer += (
                    "\n\n### 📚 Sources\n"
                )


                for i, citation in enumerate(
                    citations,
                    start=1
                ):

                    answer += (
                        f"{i}. {citation}\n"
                    )


        # ====================================================
        # SAVE ASSISTANT MESSAGE
        # ====================================================

        st.session_state[
            "messages"
        ].append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        # ====================================================
        # SAVE COMPLETE CONVERSATION
        # ====================================================

        save_current_conversation()


        # ====================================================
        # RERUN
        # ====================================================

        st.rerun()


# ============================================================
# TAB 2 — DOCUMENTS
# ============================================================

with tab_documents:

    st.subheader("📄 Documents")


    st.write(
        "### Current Document"
    )


    st.write(
        f"**File:** "
        f"{st.session_state['document_name']}"
    )

    st.write(
        f"**Pages:** "
        f"{st.session_state['page_count']}"
    )

    st.write(
        f"**Chunks:** "
        f"{len(st.session_state['chunks'])}"
    )

    st.write(
        f"**FAISS Vectors:** "
        f"{st.session_state['index'].ntotal}"
    )


    st.divider()


    st.write(
        "### 📚 Document Metadata"
    )


    for i, chunk in enumerate(
        st.session_state["chunks"],
        start=1
    ):

        source = chunk.get(
            "source",
            "Unknown"
        )

        page = chunk.get(
            "page",
            "Unknown"
        )

        section = chunk.get(
            "section"
        )


        if section:

            st.write(
                f"{i}. {source} — "
                f"Page {page} — "
                f"Section: {section}"
            )

        else:

            st.write(
                f"{i}. {source} — Page {page}"
            )


# ============================================================
# TAB 3 — SUMMARY
# ============================================================

with tab_summary:

    st.subheader(
        "📝 Document Summary"
    )


    st.write(
        "Generate an AI-powered summary "
        "of the uploaded document."
    )


    if st.button(
        "📄 Summarize Document",
        use_container_width=True
    ):

        with st.spinner(
            "🤖 Generating document summary..."
        ):

            try:

                summary = summarize_document(
                    st.session_state["chunks"],
                    batch_size=5,
                    temperature=st.session_state[
                        "settings"
                    ]["temperature"],
                    model=st.session_state[
                        "settings"
                    ]["model"]
                )


                st.session_state[
                    "summary"
                ] = summary


            except Exception as e:

                error_text = str(e)


                if (
                    "429" in error_text
                    or
                    "RESOURCE_EXHAUSTED"
                    in error_text
                    or
                    "quota"
                    in error_text.lower()
                ):

                    st.session_state[
                        "summary"
                    ] = quota_message()

                else:

                    st.session_state[
                        "summary"
                    ] = (
                        "⚠️ Unable to generate "
                        "document summary."
                    )


    if st.session_state.get(
        "summary"
    ):

        st.divider()

        st.markdown(
            st.session_state["summary"]
        )


# ============================================================
# TAB 4 — SEMANTIC SEARCH
# ============================================================

with tab_search:

    st.subheader(
        "🔎 Semantic Search"
    )


    st.write(
        "Search the uploaded document using "
        "semantic similarity."
    )


    if not st.session_state[
        "settings"
    ]["semantic_search_enabled"]:

        st.warning(
            "🔎 Semantic Search is disabled "
            "in Settings."
        )

    else:

        search_query = st.text_input(
            "Search the document:",
            placeholder=(
                "Example: machine learning applications"
            ),
            key="semantic_search_input"
        )


        search_button = st.button(
            "🔎 Search",
            use_container_width=True
        )


        if search_button and search_query.strip():

            with st.spinner(
                "🔎 Searching..."
            ):

                try:

                    search_results = retrieve_context(
                        search_query.strip(),
                        st.session_state["chunks"],
                        st.session_state["index"],
                        top_k=5,
                        distance_threshold=st.session_state[
                            "settings"
                        ]["distance_threshold"]
                    )


                except Exception as e:

                    search_results = []

                    st.error(
                        "❌ Search failed."
                    )

                    st.caption(
                        f"Technical details: {e}"
                    )


            if not search_results:

                st.warning(
                    "⚠️ No relevant information found."
                )

            else:

                st.write(
                    "### Search Results"
                )


                for i, result in enumerate(
                    search_results,
                    start=1
                ):

                    source = result.get(
                        "source",
                        "Unknown"
                    )

                    page = result.get(
                        "page",
                        "Unknown"
                    )

                    distance = result.get(
                        "distance",
                        0.0
                    )


                    st.write(
                        f"**{i}. {source}**"
                    )

                    st.write(
                        f"📄 Page: {page}"
                    )

                    st.write(
                        f"🔢 FAISS Distance: "
                        f"{distance:.4f}"
                    )

                    st.write(
                        result.get(
                            "text",
                            ""
                        )
                    )

                    st.divider()


# ============================================================
# TAB 5 — ANALYSIS
# ============================================================

with tab_analysis:

    st.subheader(
        "📊 Analysis"
    )


    # ========================================================
    # SENTIMENT
    # ========================================================

    st.write(
        "### 😊 Sentiment Analysis"
    )


    if not st.session_state[
        "settings"
    ]["sentiment_enabled"]:

        st.warning(
            "Sentiment Analysis is disabled "
            "in Settings."
        )

    elif st.session_state.get(
        "last_sentiment"
    ):

        sentiment_result = (
            st.session_state["last_sentiment"]
        )


        sentiment = sentiment_result.get(
            "sentiment",
            "Neutral"
        )


        confidence = sentiment_result.get(
            "confidence",
            0.0
        )


        st.write(
            f"**Sentiment:** {sentiment}"
        )


        st.write(
            f"**Confidence:** "
            f"{confidence:.2f}%"
        )


    else:

        st.info(
            "Ask a question in the Chat tab "
            "to see sentiment analysis."
        )


    st.divider()


    # ========================================================
    # INTENT
    # ========================================================

    st.write(
        "### 🎯 Intent Analysis"
    )


    if not st.session_state[
        "settings"
    ]["intent_enabled"]:

        st.warning(
            "Intent Analysis is disabled "
            "in Settings."
        )

    elif st.session_state.get(
        "last_intent"
    ):

        intent_result = (
            st.session_state["last_intent"]
        )


        intent = intent_result.get(
            "intent",
            "UNKNOWN"
        )


        confidence = intent_result.get(
            "confidence",
            0.0
        )


        st.write(
            f"**Intent:** {intent}"
        )


        st.write(
            f"**Confidence:** "
            f"{confidence:.2f}%"
        )


    else:

        st.info(
            "Ask a question in the Chat tab "
            "to see intent analysis."
        )


# ============================================================
# TAB 6 — SETTINGS
# ============================================================

with tab_settings:

    st.subheader(
        "⚙ Settings"
    )


    # ========================================================
    # AI CONFIGURATION
    # ========================================================

    st.write(
        "### 🤖 Model"
    )


    model_options = [
        "gemini-3.6-flash"
    ]


    current_model = st.session_state[
        "settings"
    ]["model"]


    selected_model = st.selectbox(
        "AI Model",
        model_options,
        index=(
            model_options.index(
                current_model
            )
            if current_model in model_options
            else 0
        )
    )


    st.session_state[
        "settings"
    ]["model"] = selected_model


    # ========================================================
    # TEMPERATURE
    # ========================================================

    st.write(
        "### 🌡 Temperature"
    )


    temperature = st.slider(
        "Response creativity",
        min_value=0.0,
        max_value=1.0,
        value=float(
            st.session_state[
                "settings"
            ]["temperature"]
        ),
        step=0.1
    )


    st.session_state[
        "settings"
    ]["temperature"] = temperature


    # ========================================================
    # TOP-K
    # ========================================================

    st.write(
        "### 🔢 Top-K"
    )


    top_k = st.number_input(
        "Number of retrieved chunks",
        min_value=1,
        max_value=20,
        value=int(
            st.session_state[
                "settings"
            ]["top_k"]
        ),
        step=1
    )


    st.session_state[
        "settings"
    ]["top_k"] = top_k


    # ========================================================
    # DISTANCE THRESHOLD
    # ========================================================

    st.write(
        "### 🎯 Distance Threshold"
    )


    distance_threshold = st.number_input(
        "Maximum FAISS distance",
        min_value=0.0,
        max_value=10.0,
        value=float(
            st.session_state[
                "settings"
            ]["distance_threshold"]
        ),
        step=0.05,
        format="%.2f"
    )


    st.session_state[
        "settings"
    ]["distance_threshold"] = (
        distance_threshold
    )


    st.divider()


    # ========================================================
    # FEATURES
    # ========================================================

    st.write(
        "### 🔧 Features"
    )


    chat_history_enabled = st.checkbox(
        "💬 Chat History",
        value=st.session_state[
            "settings"
        ]["chat_history_enabled"]
    )


    sentiment_enabled = st.checkbox(
        "📊 Sentiment Analysis",
        value=st.session_state[
            "settings"
        ]["sentiment_enabled"]
    )


    intent_enabled = st.checkbox(
        "🎯 Intent Analysis",
        value=st.session_state[
            "settings"
        ]["intent_enabled"]
    )


    citations_enabled = st.checkbox(
        "📚 Source Citations",
        value=st.session_state[
            "settings"
        ]["source_citations_enabled"]
    )


    semantic_enabled = st.checkbox(
        "🔎 Semantic Search",
        value=st.session_state[
            "settings"
        ]["semantic_search_enabled"]
    )


    st.session_state[
        "settings"
    ]["chat_history_enabled"] = (
        chat_history_enabled
    )


    st.session_state[
        "settings"
    ]["sentiment_enabled"] = (
        sentiment_enabled
    )


    st.session_state[
        "settings"
    ]["intent_enabled"] = (
        intent_enabled
    )


    st.session_state[
        "settings"
    ]["source_citations_enabled"] = (
        citations_enabled
    )


    st.session_state[
        "settings"
    ]["semantic_search_enabled"] = (
        semantic_enabled
    )


    st.divider()


    # ========================================================
    # CURRENT CONFIGURATION
    # ========================================================

    st.write(
        "### 📋 Current Configuration"
    )


    st.write(
        f"**Model:** "
        f"{st.session_state['settings']['model']}"
    )


    st.write(
        f"**Temperature:** "
        f"{st.session_state['settings']['temperature']:.1f}"
    )


    st.write(
        f"**Top-K:** "
        f"{st.session_state['settings']['top_k']}"
    )


    st.write(
        f"**Distance Threshold:** "
        f"{st.session_state['settings']['distance_threshold']:.2f}"
    )


    st.write(
        f"**Chat History:** "
        f"{'Enabled' if chat_history_enabled else 'Disabled'}"
    )


    st.write(
        f"**Sentiment Analysis:** "
        f"{'Enabled' if sentiment_enabled else 'Disabled'}"
    )


    st.write(
        f"**Intent Analysis:** "
        f"{'Enabled' if intent_enabled else 'Disabled'}"
    )


    st.write(
        f"**Source Citations:** "
        f"{'Enabled' if citations_enabled else 'Disabled'}"
    )


    st.write(
        f"**Semantic Search:** "
        f"{'Enabled' if semantic_enabled else 'Disabled'}"
    )


    st.divider()


    # ========================================================
    # CLEAR CURRENT CHAT
    # ========================================================

    if st.button(
        "🗑 Clear Chat",
        use_container_width=True
    ):

        clear_current_chat()

        st.success(
            "✅ Current chat cleared."
        )

        st.rerun()


    # ========================================================
    # CLEAR ALL HISTORY
    # ========================================================

    if st.button(
        "🗑 Clear Chat History",
        use_container_width=True
    ):

        clear_all_history()

        st.success(
            "✅ All chat history cleared."
        )

        st.rerun()
