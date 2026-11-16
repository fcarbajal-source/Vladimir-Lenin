import re
import streamlit as st
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
MODEL_ID = "gemini-3.5-flash-lite"
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 280
MIN_WORDS = 15
MAX_WORDS = 120
MAX_REMEMBERED_QUESTIONS = 6

SYSTEM_INSTRUCTION = """
You are Vladimir Ilyich Lenin (born 22 April 1870 – died 21 January 1924), leader of the Bolshevik Party and Chairman of the Council of People's Commissars of Soviet Russia.

AUDIENCE & VOICE:
- Speak directly to U.S. 10th-grade high school students (ages 15–16).
- Sound like Vladimir Lenin himself: urgent, analytical, sharp, and persuasive—like a revolutionary orator addressing a hall (e.g., "Young student," "Let us examine the concrete facts," "History does not wait").
- Keep sentences clear and understandable for a 10th-grade American student. Briefly explain terms like "Soviets" (workers' and soldiers' councils), "Duma" (Russian parliament), or "kulaks" (wealthy peasants) when used.

MANDATED HISTORICAL ANCHORS (BE FACT-DENSE):
Anchor every answer in verified dates, people, letters, decrees, speeches, and legal/parliamentary records—including critical archival records:
- Early Exile & Theory (1895–1917): Arrested in St. Petersburg (1895) and exiled to Siberia; What Is to Be Done? (1902); 1903 Bolshevik-Menshevik split; 1905 Revolution; April Theses (April 1917) calling for "All power to the Soviets!" and State Duma opposition.
- October Revolution & First Laws (1917–1918): Letters to the Bolshevik Central Committee (Sept–Oct 1917) urging insurrection; Second All-Russia Congress of Soviets adopting the Decree on Peace and Decree on Land (8 Nov 1917 / 26 Oct O.S.); Dissolution of the Constituent Assembly (Jan 1918) and Declaration of Rights of the Working and Exploited People; Treaty of Brest-Litovsk (3 March 1918); 1918 RSFSR Constitution.
- Letters, Civil War & U.S. Congressional/Court Records (1918–1920): Letter to American Workers (20 Aug 1918) appealing to U.S. workers during Allied intervention; Aug 11, 1918 telegram to Penza communists ordering ruthless suppression of kulak uprisings; Sept 15, 1919 letter to Maxim Gorky regarding arrests of intellectuals; March 1919 gramophone speeches ("What Is Soviet Power?"); U.S. Senate Overman Committee hearings/report on Bolshevik propaganda (Feb–June 1919); U.S. Supreme Court Red Scare rulings reacting to Bolshevik agitation (Abrams v. United States, 1919).
- NEP, USSR & Final Letters (1921–1924): 10th Party Congress introducing the New Economic Policy (NEP) and "Tax in Kind" (March 1921); Treaty on the Creation of the USSR (30 Dec 1922); confidential Letter to the Congress / "Testament" (Dec 1922–Jan 1923) warning about Joseph Stalin's rudeness and rivalry with Leon Trotsky.

MANDATORY SOURCES & CITATIONS:
Ground every response in primary and secondary sources—especially letters written to and from Vladimir Lenin, official government decrees, speeches, Russian parliamentary (Duma/Constituent Assembly) or U.S. Congressional documents, court case decisions/records, and legislation.
Every response MUST end with a brief citation naming the specific document/letter and at least one of these approved digital archives:
1. Library of Congress Digital Collections
2. HathiTrust Digital Library
3. Lenin Internet Archive (Marxists Internet Archive)
4. V.I. Lenin Internet Library (Marx2Mao)
5. Full Collection of Lenin's Works (Materialisticheskaia Dialektika)
6. Lenin's Archive (Marxist-Leninists.org)
7. Leaders of the Soviet Period: Lenin (Federal Archival Agency of Russia)
8. Library of Congress: Revelations from the Russian Archives
9. Harvard Library: Digital Handbook for Research on Soviet History
10. Internet Archive: Lenin Collected Works Collection
11. Kennan Institute Russian History Audio Archive (Wilson Center)
12. Internet History Sourcebooks Project: Russian Revolution (Fordham University)

STRICT LENGTH & ACCURACY RULES:
- Every reply MUST be between 15 words minimum and 120 words maximum (target 65–105 words), INCLUDING the final "Source:" citation.
- Pack in real dates, names, and primary sources. Never invent facts or citations. Do not sanitize violent or controversial orders found in the archives—present the documented historical record objectively in Lenin's voice. If asked about events after January 1924, state plainly that your life ended in 1924 while still citing an archive.
""".strip()


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
@st.cache_resource
def get_genai_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def enforce_word_limits(text: str, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> str:
    """Guarantees response stays strictly between 15 and 120 words."""
    cleaned = " ".join(text.split())
    words = cleaned.split()

    if len(words) > max_words:
        source_match = re.search(r"(Source:\s*.+)$", cleaned)
        source_tail = source_match.group(1) if source_match else ""
        source_words = source_tail.split() if source_tail else []

        if source_tail and len(source_words) < max_words - 15:
            body_text = cleaned[: source_match.start()].strip()
            budget = max_words - len(source_words)
            trimmed_body = " ".join(body_text.split()[:budget])
            last_punct = max(trimmed_body.rfind("."), trimmed_body.rfind("!"), trimmed_body.rfind("?"))
            if last_punct > 20:
                trimmed_body = trimmed_body[: last_punct + 1]
            cleaned = f"{trimmed_body} {source_tail}".strip()
        else:
            trimmed = " ".join(words[:max_words])
            last_punct = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"))
            cleaned = trimmed[: last_punct + 1] if last_punct > 20 else trimmed + "..."

    elif len(words) < min_words:
        cleaned = (
            f"{cleaned} Examine the decrees and letters from 1917 to 1923 to judge the concrete facts. "
            "Source: Lenin Internet Archive (Marxists Internet Archive)."
        ).strip()

    return cleaned


def build_limited_history(messages: list[dict], new_prompt: str) -> list[types.Content]:
    """
    Keeps ONLY the previous 6 user questions (and their 6 model answers)
    plus the current user question.
    """
    prior_turns = messages[-(MAX_REMEMBERED_QUESTIONS * 2) :]
    contents = []
    for msg in prior_turns:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])],
            )
        )
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=new_prompt)],
        )
    )
    return contents


def ask_vladimir_lenin(client: genai.Client, history_contents: list[types.Content]) -> str:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    )
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=history_contents,
        config=config,
    )
    raw_text = (response.text or "").strip()
    return enforce_word_limits(raw_text)


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ask Vladimir Lenin (1870–1924)",
    page_icon="📜",
    layout="centered",
)

st.title("📜 Speak with Vladimir Lenin")
st.caption(
    "An interactive primary-source history chatbot for U.S. 10th-grade students "
    f"• Powered by `{MODEL_ID}`"
)

# Load hidden API key from Streamlit Secrets
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error(
        "Missing `GEMINI_API_KEY`. Add it to `.streamlit/secrets.toml` locally "
        "or in Streamlit Community Cloud → App Settings → Secrets."
    )
    st.stop()

client = get_genai_client(api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls & archive reference
with st.sidebar:
    st.header("Historical Guidelines")
    remembered_q_count = min(
        MAX_REMEMBERED_QUESTIONS,
        sum(1 for m in st.session_state.messages if m["role"] == "user"),
    )
    st.metric("Active Memory (Max 6 Questions)", f"{remembered_q_count} / {MAX_REMEMBERED_QUESTIONS}")
    st.write("**Reading Level:** U.S. 10th Grade")
    st.write("**Response Length:** 15–120 words")
    st.write(f"**Temperature:** {TEMPERATURE} | **Max Tokens:** {MAX_OUTPUT_TOKENS}")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Try Asking:")
    sample_questions = [
        "What did you write in your August 1918 Letter to American Workers?",
        "Why did you dissolve the Constituent Assembly in January 1918?",
        "What did your 1918 Penza telegram and 1919 letter to Gorky reveal about Civil War repression?",
        "How did the U.S. Senate Overman Committee investigate Bolshevism in 1919?",
        "What warnings did you give about Stalin in your 1922–1923 Letter to the Congress?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_prompt = q

    st.divider()
    st.subheader("Referenced Digital Archives")
    st.markdown(
        "- Library of Congress Digital Collections\n"
        "- HathiTrust Digital Library\n"
        "- Lenin Internet Archive (Marxists Internet Archive)\n"
        "- V.I. Lenin Internet Library (Marx2Mao)\n"
        "- Full Collection of Lenin's Works (Materialisticheskaia Dialektika)\n"
        "- Lenin's Archive (Marxist-Leninists.org)\n"
        "- Leaders of the Soviet Period: Lenin (Federal Archival Agency of Russia)\n"
        "- Library of Congress: Revelations from the Russian Archives\n"
        "- Harvard Library: Digital Handbook for Research on Soviet History\n"
        "- Internet Archive: Lenin Collected Works Collection\n"
        "- Kennan Institute Russian History Audio Archive (Wilson Center)\n"
        "- Internet History Sourcebooks Project: Russian Revolution (Fordham University)"
    )

# Render chat history
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="📜"):
        st.write(
            "Greetings, young student. I am Vladimir Lenin, speaking from the records of 1917 to 1924. "
            "Ask Me about the Decree on Peace, My 1918 Letter to American Workers, Soviet laws, or "
            "My final letters on party leadership—let us examine the primary documents together. "
            "Source: Library of Congress: Revelations from the Russian Archives."
        )

for msg in st.session_state.messages:
    avatar = "📜" if msg["role"] == "assistant" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Accept chat input or sidebar button prompt
prompt = st.chat_input("Ask Vladimir Lenin a historical question...")
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    with st.chat_message("user", avatar="🎓"):
        st.write(prompt)

    # Build history with ONLY the previous 6 questions + current prompt
    history_contents = build_limited_history(st.session_state.messages, prompt)

    with st.chat_message("assistant", avatar="📜"):
        with st.spinner("Consulting the Soviet & Library of Congress Archives..."):
            try:
                reply = ask_vladimir_lenin(client, history_contents)
            except Exception as exc:
                st.error(f"Could not reach Gemini API: {exc}")
                st.stop()
        st.write(reply)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": reply})
