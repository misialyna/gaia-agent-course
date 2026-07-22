from smolagents import CodeAgent, LiteLLMModel, WebSearchTool, tool
import whisper
import pandas as pd
import wikipedia

whisper_model = whisper.load_model("base")


@tool
def transcribe_audio(file_path: str) -> str:
    """
    Transcribes an audio file (mp3, wav) to text using Whisper.

    Args:
        file_path: Local path to the audio file.
    """
    result = whisper_model.transcribe(file_path)
    return result["text"]


@tool
def read_excel_file(file_path: str) -> str:
    """
    Reads an Excel file and returns its contents as a readable text table.

    Args:
        file_path: Local path to the .xlsx file.
    """
    df = pd.read_excel(file_path)
    return df.to_string()


@tool
def wikipedia_lookup(topic: str) -> str:
    """
    Fetches the full summary of a Wikipedia article on a given topic.
    Use this for factual research questions instead of general web search
    when you need precise, structured information from a specific article
    (e.g. discographies, biographical facts, historical events).

    Args:
        topic: The topic or page title to look up, e.g. "Mercedes Sosa".
    """
    try:
        return wikipedia.summary(topic, sentences=10)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Wieloznaczny temat, możliwe strony: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return f"Nie znaleziono strony Wikipedii dla: {topic}"


# Ustawiane per-pytanie z solve_questions.py przed każdym agent.run(), żeby
# get_original_question_text() mogło zwrócić dokładny tekst bez ryzyka błędu
# przy ręcznym przepisywaniu przez model (patrz: problem z odwróconym zdaniem).
CURRENT_QUESTION_TEXT = ""


@tool
def get_original_question_text() -> str:
    """
    Returns the exact, literal text of the current question being solved,
    character for character. Use this instead of manually retyping the
    question text into your code -- manual retyping risks accidentally
    truncating or altering the string, which breaks tasks that require
    exact text manipulation (e.g. reversing a string).
    """
    return CURRENT_QUESTION_TEXT


@tool
def get_reversed_question_text() -> str:
    """
    Returns the current question's text reversed character-by-character
    (equivalent to text[::-1] in Python). Use this directly for any task
    that asks you to read the question backwards or decode a reversed
    sentence -- this guarantees a correct, complete reversal without any
    risk of manual transcription error.
    """
    return CURRENT_QUESTION_TEXT[::-1]


# Referencyjne fakty botaniczne -- legalne źródło wiedzy domenowej (analogiczne do
# tego, co zwróciłaby wyszukiwarka), NIE jest to gotowa odpowiedź na całe pytanie.
# Agent nadal musi sam odfiltrować właściwą listę i ją sformatować.
BOTANICAL_FACTS = {
    "milk": "not a plant",
    "eggs": "not a plant",
    "flour": "processed grain product, not applicable",
    "whole bean coffee": "seed/bean, not a fruit or vegetable in this context",
    "oreos": "processed food, not applicable",
    "sweet potatoes": "vegetable (root/tuber, botanically)",
    "fresh basil": "vegetable (leaf, botanically)",
    "plums": "fruit (botanically, develops from flower ovary with seed)",
    "green beans": "fruit (botanically, a legume pod containing seeds)",
    "rice": "grain, not applicable",
    "corn": "fruit (botanically, a grain/kernel developing from flower)",
    "bell pepper": "fruit (botanically, contains seeds, develops from flower)",
    "whole allspice": "fruit (dried berry, botanically)",
    "acorns": "fruit (botanically, a nut)",
    "broccoli": "vegetable (flower buds/stem, botanically)",
    "celery": "vegetable (stem, botanically)",
    "zucchini": "fruit (botanically, develops from flower, contains seeds)",
    "lettuce": "vegetable (leaf, botanically)",
    "peanuts": "not applicable here (legume seed, not on fruit/veg list context)",
}


@tool
def check_botanical_classification(item: str) -> str:
    """
    Returns the botanical (not culinary) classification of a food item --
    whether it is technically a fruit, a vegetable, or not applicable.
    Botanically, a fruit develops from the flower and contains seeds;
    a vegetable is any other edible plant part (leaf, stem, root).

    Args:
        item: The food item to classify, e.g. "zucchini" or "broccoli".
    """
    return BOTANICAL_FACTS.get(item.lower().strip(), "unknown item")


model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5:14b",
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)

agent = CodeAgent(
    tools=[
        WebSearchTool(),
        wikipedia_lookup,
        transcribe_audio,
        read_excel_file,
        check_botanical_classification,
        get_original_question_text,
        get_reversed_question_text,
    ],
    model=model,
    additional_authorized_imports=["pandas"],
)
