from typing import TypedDict, List, Optional, Annotated
import operator
import os
from pathlib import Path
import requests
from datetime import date
import re
from urllib.parse import unquote
import uuid
from dotenv import load_dotenv
import traceback
import subprocess
import sys
import base64
import logging
import chess
import chess.engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

# --- Langchain és Google/Gemini specifikus importok ---
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmCategory,
    HarmBlockThreshold,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
#from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # MessagesPlaceholder nem használt még, de hasznos lehet
from langchain_core.tools import BaseTool
# Pydantic import a Langchain figyelmeztetésének megfelelően (v1 kompatibilitás)
from pydantic.v1 import BaseModel, Field

# --- LangGraph importok ---
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.sqlite import SqliteSaver # Egyelőre nem használjuk

# --- Egyéb importok (pl. eszközökhöz) ---
from tavily import TavilyClient # Csak ha Tavily-t használsz
import google.generativeai as genai
from google.generativeai import types as GoogleGenAITypes
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch # <<< EZ A SOR KULCSFONTOSSÁGÚ!
#from external_google_search_EXACT import search_google_exact_original

# === Konstansok ===
SCORING_API_BASE_URL = "https://agents-course-unit4-scoring.hf.space"
TEMP_DIR = Path("./temp_gaia_files")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
CURRENT_DATE_STR = date.today().isoformat()
DEFAULT_MAX_SEARCH_SUB_ITERATIONS = 3

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    # Ha nincs a .env-ben vagy a rendszerben, itt döntheted el, mi történjen:
    # - Kérdezd meg a felhasználót (mint az eredeti kódnál, de ez interaktívvá teszi)
    # google_api_key = getpass.getpass("Enter your Google AI API key: ")
    # - Dobj hibát és lépj ki, jelezve, hogy a .env fájl hiányos
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please add it to your .env file.")

os.environ["GOOGLE_API_KEY"] = google_api_key
print("Google API key loaded.")

"""tavily_api_key = os.getenv("TAVILY_API_KEY")

if not tavily_api_key:
# Ha nincs a .env-ben vagy a rendszerben
raise ValueError("TAVILY_API_KEY environment variable not set. Please add it to your .env file.")

# Beállítjuk az os.environ-ban is, ha a tool úgy keresi
os.environ["TAVILY_API_KEY"] = tavily_api_key
print("Tavily API key loaded.")"""


# === LLM Inicializálása ===
MODEL_NAME="gemini-2.5-flash-preview-05-20"

llm = None
try:
    llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.15,
    #max_tokens=None,
    #timeout=180,
    #max_retries=2,
    # convert_system_message_to_human=True # Régebbi Gemini modelleknél szükség lehetett erre,
    # ha a SystemMessage-et nem kezelték jól.
    # Az újabb modellek (pl. 1.5 Pro, Flash) jobban kezelik.
    # Teszteld, hogy nélküle is jól működik-e a SystemMessage.
    safety_settings={ # Opcionális, ha a safety filterek túl agresszívak lennének ()
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    )
    print(f"Google Gemini LLM ({MODEL_NAME}) initialized successfully.")
except Exception as e:
    print(f"Could not initialize Google Gemini LLM. Error: {e}")
    print(
        "Please ensure your GOOGLE_API_KEY is set correctly and you have the 'langchain-google-genai' package installed.")
    print("Falling back to no LLM for planning (placeholder logic will be used).")

# === Agent Állapot Definiálása ===
class AgentState(TypedDict):
    original_question: str
    task_id: str
    # A kérdéshez tartozó letöltött fájl elérési útja, ha van
    file_path: Optional[str]
    # Az LLM által generált terv vagy következő lépés
    plan: Optional[List[str]]
    # Az eddig összegyűjtött információk, válaszok az eszközöktől
    intermediate_steps: Annotated[List[tuple], operator.add]
    # A webes keresés eredménye
    # search_results: Optional[str]
    # A kód futtatásának eredménye
    # code_output: Optional[str]
    # A hangfájl átirata
    # transcript: Optional[str]
    # Az aktuális próbálkozások száma (pl. hibák után)
    iterations: int
    # A végső válasz, amit az agent adni kíván
    final_answer: Optional[str]
    # Hibaüzenet, ha valami probléma lépett fel
    error_message: Optional[str]
    # Az aktuálisan használni kívánt eszköz neve
    current_tool_name: Optional[str]
    # Az aktuális eszközhöz tartozó bemenet
    current_tool_input: Optional[dict]

    # --- ÚJ: Excel-specifikus al-ág állapotai ---
    excel_processing_active: bool
    excel_original_question: Optional[str]
    excel_file_to_process: Optional[str]
    excel_current_pandas_code: Optional[str]
    excel_code_execution_history: Annotated[List[tuple], operator.add]  # [(code_string, output_string), ...]
    excel_processing_status: Optional[str]  # pl. "starting", "inspecting", "calculating", "error_code", "success_code", "final_answer_ready", "max_iterations_reached", "failed_by_llm"
    excel_iteration_count: int
    excel_max_sub_iterations: int  # Maximális iterációk az Excel al-ágon belül
    excel_final_result_from_sub_agent: Optional[str]  # Az Excel al-ág végeredménye

    # === ÚJ: Keresési al-ág állapotai ===
    search_sub_graph_active: bool
    search_original_query_from_main_llm: Optional[str]
    search_current_refined_query: Optional[str]
    search_results_history: Annotated[List[tuple], operator.add]  # Lista: [(query1, result1_str), (query2, result2_str), ...]
    search_analysis_history: Annotated[List[tuple], operator.add]  # Lista: [(search_output_for_analysis, analysis_text_from_llm), ...]
    search_processing_status: Optional[str]
    search_iteration_count: int
    search_max_sub_iterations: int
    search_final_summary_for_main_agent: Optional[str]

# === Eszközök Definiálása ===

# --- PythonInterpreterTool ---

class PythonInterpreterInput(BaseModel):
    code_string: Optional[str] = Field(
        default=None,
        description="A string containing the Python code to execute directly. Use this if you have the code as a string."
    )
    file_path: Optional[str] = Field(
        default=None,
        description="The local file path to the Python script to be executed. Use this if 'code_string' is not provided and the code is in a file."
    )
    timeout_seconds: int = Field(
        default=30,
        description="Maximum execution time for the script in seconds."
    )

    # Megfontolható további paraméterek:
    # args: Optional[List[str]] = Field(default_factory=list, description="Arguments to pass to the script if it accepts them via sys.argv. Note: This is an advanced feature and might require specific script handling.")
    # timeout: int = Field(default=10, description="Timeout in seconds for the script execution. Note: Not implemented in this basic version.")


class PythonInterpreterTool(BaseTool):
    name: str = "python_interpreter"
    description: str = (
        "Executes Python code in a separate process and returns its standard output (stdout) and standard error (stderr). "
        "You can provide the code directly as a string using the 'code_string' argument, OR "
        "provide a 'file_path' to a Python script. Prefer 'code_string' for short, self-contained snippets, "
        "and 'file_path' for longer scripts or those already downloaded. "
        "If the script exceeds the specified timeout (default 10 seconds), it will be terminated. "
        "Use this tool when a Python file needs to be run or when you need to execute a Python code snippet. "
        "SECURITY WARNING: This tool executes Python code. Ensure the code is trusted."
    )
    args_schema: type[BaseModel] = PythonInterpreterInput

    def _run(self,
             code_string: Optional[str] = None,
             file_path: Optional[str] = None,
             timeout_seconds: int = 30) -> str:  # Alapértelmezett timeout itt is 30s

        python_code_to_execute = ""
        source_description = ""
        # Ideiglenes fájl a code_string-hez vagy a diagnosztikával kiegészített fájlhoz
        temp_script_to_run = None
        # Annak jelzése, hogy a temp_script_to_run-t a végén törölni kell-e
        # (csak akkor, ha mi hoztuk létre a code_string-hez vagy a diagnosztikához)
        cleanup_temp_script = False

        # --- Diagnosztikai kód definíciója ---
        diagnostic_code = """import sys
import os
print(f"--- Python Subprocess Diagnostics (PythonInterpreterTool) ---")
print(f"Python Executable: {sys.executable}")
print("Python Version:", sys.version.replace('\\n', ' '))
print(f"Current Working Directory: {os.getcwd()}")
print(f"sys.path: {sys.path}")
try:
    import chess
    print(f"python-chess module found at (or is built-in): {getattr(chess, '__file__', 'N/A (location not available)')}")
except ImportError:
    print("python-chess module NOT FOUND by this subprocess!")
except Exception as e_diag:
    print(f"Error during chess import test in diagnostics: {str(e_diag)}")
print(f"--- End Diagnostics ---")
print("# --- Original Code Starts Below (if any) ---")
"""
        # --- Diagnosztikai kód vége ---

        if code_string:
            python_code_to_execute = code_string
            source_description = "direct code string"
            print(f"  Attempting to execute Python code from direct string input (with diagnostics).")
            try:
                final_code_to_write = (
                        diagnostic_code.strip() +  # Esetleges felesleges kezdő/záró whitespace-ek eltávolítása
                        "\n\n# --- User/LLM Code Starts Below ---\n" +
                        python_code_to_execute.strip()  # Az LLM kódjáról is eltávolítjuk a felesleges whitespace-eket
                )
                temp_script_path_obj = TEMP_DIR / f"temp_script_diag_{uuid.uuid4()}.py"
                with open(temp_script_path_obj, "w", encoding="utf-8") as f:
                    f.write(final_code_to_write)
                temp_script_to_run = str(temp_script_path_obj)
                cleanup_temp_script = True  # Ezt mi hoztuk létre, törölni kell
            except Exception as e:
                return f"Error creating temporary script file for code_string with diagnostics: {e}"

        elif file_path:
            source_description = f"file: {file_path}"
            print(f"  Attempting to execute Python script from: {file_path}")
            if not os.path.exists(file_path):
                return f"Error: Python script not found at path: {file_path}."

            # Döntés: A file_path ágon is beillesszük a diagnosztikát?
            # Igen, most már ezt is megcsináljuk, hogy konzisztensebb legyen.
            print(f"  Prepending diagnostics to script from: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as f_orig:
                    original_code_from_file = f_orig.read()

                final_code_to_write = (
                        diagnostic_code.strip() +
                        "\n\n# --- Original Code from File Starts Below ---\n" +
                        original_code_from_file  # Itt nem strip()-elünk, mert a fájl formázása lehet szándékos
                )

                temp_script_path_obj = TEMP_DIR / f"temp_script_diag_file_{uuid.uuid4()}.py"
                with open(temp_script_path_obj, "w", encoding="utf-8") as f_temp:
                    f_temp.write(final_code_to_write)

                temp_script_to_run = str(temp_script_path_obj)
                cleanup_temp_script = True  # Ezt is mi hoztuk létre, törölni kell
                source_description = f"file (with diagnostics wrapper): {file_path}"  # Jelezzük, hogy módosult
            except Exception as e:
                return f"Error preparing diagnostic script wrapper for file {file_path}: {e}"
        else:
            return "Error: No Python code provided to 'python_interpreter'. Either 'code_string' or 'file_path' argument must be specified."

        print(
            f"  Executing Python code ({source_description}) using subprocess. Timeout: {timeout_seconds}s. Script to run: {temp_script_to_run}")

        python_executable = sys.executable

        try:
            process = subprocess.run(
                [python_executable, temp_script_to_run],
                # Mindig a (potenciálisan diagnosztikával ellátott) temp scriptet futtatjuk
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False
            )

            output = ""
            if process.stdout:
                output += f"Standard Output:\n{process.stdout.strip()}\n"
            if process.stderr:
                output += f"Standard Error (if any):\n{process.stderr.strip()}\n"

            if not output:
                output = "Python script executed. No output produced on stdout or stderr."

            # A visszatérési kódot is érdemes az LLM tudomására hozni, ha nem nulla
            if process.returncode != 0:
                # Hozzáadjuk a visszatérési kódot az outputhoz, hogy az LLM lássa
                output_with_rc = f"{output.strip()}\nScript exited with return code: {process.returncode}."
                # A "Python code ... executed with errors." üzenet helyett adjuk vissza a teljes outputot a hibakóddal.
                # Az LLM-nek kell eldöntenie, hogy ez hiba-e a kontextus alapján.
                # A plan_next_step promptja majd instruálhatja, hogyan értelmezze a nem nulla visszatérési kódokat.
                print(
                    f"  Python script exited with non-zero return code ({process.returncode}). Output including RC: {output_with_rc[:300]}...")
                return output_with_rc  # Teljes output + visszatérési kód

            # Ha a visszatérési kód 0, akkor csak az outputot adjuk vissza
            print(f"  Python script executed successfully (return code 0). Output: {output[:300]}...")
            return output  # Csak a stdout/stderr, ha a visszatérési kód 0

        except subprocess.TimeoutExpired:
            return (f"Error: Python script from {source_description} (using {temp_script_to_run}) "
                    f"exceeded timeout of {timeout_seconds} seconds. Execution was terminated.")
        except Exception as e:
            tb_str = traceback.format_exc()
            return (
                f"Error executing Python script from {source_description} (using {temp_script_to_run}) via subprocess: "
                f"{type(e).__name__} - {e}\nFull Traceback:\n{tb_str}")
        finally:
            if cleanup_temp_script and temp_script_to_run and os.path.exists(temp_script_to_run):
                try:
                    os.remove(temp_script_to_run)
                    print(f"  Temporary script file {temp_script_to_run} removed.")
                except Exception as e:
                    print(f"  Error removing temporary script file {temp_script_to_run}: {e}")

# --- FileDownloaderTool ---
class FileDownloaderInput(BaseModel):
    task_id: str = Field(description="The unique identifier for the task, used to construct the file download URL.")

class FileDownloaderTool(BaseTool):
    name: str = "file_downloader"
    description: str = (
        "Downloads a file associated with a given task_id from the GAIA scoring API. "
        "The file is saved locally to a temporary directory. "
        "Use this tool if the user's question mentions an 'attached file', 'image', 'video', 'audio', 'code', 'excel' or similar, "
        "and you suspect a file needs to be fetched using its task_id to answer the question."
    )
    args_schema: type[BaseModel] = FileDownloaderInput

    def _run(self, task_id: str) -> str:
        file_url = f"{SCORING_API_BASE_URL}/files/{task_id}"
        # local_file_name = f"{task_id}_downloaded_file" # Kezdetben egy általános név
        local_file_path_str = ""  # Visszatérési értékhez

        print(f"Attempting to download file from: {file_url}")
        try:
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()

            content_disposition = response.headers.get('Content-Disposition')
            derived_filename = f"{task_id}_downloaded_file"  # Alapértelmezett, ha nincs jobb
            if content_disposition:
                fname_match = re.findall(r"filename\*?=([^;]+)", content_disposition)
                if fname_match:
                    potential_fname = fname_match[0].strip().strip('"').strip("'")
                    if potential_fname.lower().startswith("utf-8''"):
                        potential_fname = unquote(potential_fname[len("utf-8''"):])
                    else:
                        potential_fname = unquote(potential_fname)

                    safe_fname = "".join(c for c in potential_fname if c.isalnum() or c in ('.', '_', '-'))
                    if safe_fname:
                        derived_filename = f"{task_id}_{safe_fname}"

            local_file_path = TEMP_DIR / derived_filename
            local_file_path_str = str(local_file_path)

            with open(local_file_path, "wb") as f:
                f.write(response.content)
            print(f"File downloaded successfully to: {local_file_path_str}")
            return f"File downloaded successfully. Available at path: {local_file_path_str}"
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 404:
                return f"Error: File not found for task_id {task_id} (404). No file might be associated with this task. Do not try to download this file again for this task_id."
            return f"Error downloading file for task_id {task_id}: HTTP {response.status_code} - {http_err}. Response: {response.text[:200]}"
        except requests.exceptions.RequestException as req_err:
            return f"Error downloading file for task_id {task_id}: {req_err}"
        except Exception as e:
            return f"An unexpected error occurred while downloading file for task_id {task_id}: {str(e)}"


# --- TavilyWebSearchTool ---
"""class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to find information on the web.")


class TavilyWebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Performs a web search using Tavily to find relevant information for a given query. "
        "Returns a concise answer based on search results, or a list of search result snippets if a detailed answer is not found. "
        "Use this tool when you need to find information online, such as facts, definitions, current events, or details not present in prior context."
    )
    args_schema: type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        tavily_api_key_env = os.getenv("TAVILY_API_KEY")  # Próbáljuk környezeti változóból
        if not tavily_api_key_env:  # Ha nincs környezeti változó, használjuk a kódban definiáltat (ha van)
            tavily_api_key_env = tavily_api_key if 'TAVILY_API_KEY' in globals() and tavily_api_key else None

        if not tavily_api_key_env:
            return "Error: TAVILY_API_KEY not set in environment or code. Web search is unavailable."

        client = TavilyClient(api_key=tavily_api_key_env)
        print(f"Performing Tavily web search for: {query}")
        try:
            response = client.search(query=query, search_depth="basic", max_results=3)  # Kevesebb eredmény elég lehet
            if "answer" in response and response["answer"]:
                return f"Tavily Answer: {response['answer']}"
            if "results" in response and response["results"]:
                formatted_results = []
                for res in response["results"]:
                    formatted_results.append(
                        f"Title: {res.get('title', 'N/A')}\nURL: {res.get('url', 'N/A')}\nSnippet: {res.get('content', 'N/A')[:250]}...")
                return f"Tavily Search Results (snippets):\n" + "\n\n".join(formatted_results)
            return "No direct answer or detailed results found by Tavily."
        except Exception as e:
            return f"Error during Tavily search: {str(e)}"
"""

class MultimodalInput(BaseModel):
    file_path: str = Field(
        description="The local path to the multimodal file (image, audio, video) after it has been downloaded by 'file_downloader'.")
    user_prompt: str = Field(
        description="The specific question or instruction for the LLM on what to do with the multimodal file. E.g., 'Transcribe the audio.', 'Describe this image.'")
    mime_type: Optional[str] = Field(default=None,
                                     description="The MIME type of the file, e.g., 'audio/mpeg', 'image/png', 'video/mp4'. If not provided, it will be guessed from the file extension.")


class MultimodalProcessingTool(BaseTool):
    name: str = "multimodal_file_processor"  # Adjunk neki egyértelmű nevet
    description: str = (
        "Processes a previously downloaded multimodal file (audio, image, or video) using the main LLM's capabilities. "
        "Use this tool AFTER a relevant file has been downloaded using 'file_downloader'. "
        "You MUST provide the 'file_path' of the downloaded file and a 'user_prompt' describing what to do with it. "
        "Providing the 'mime_type' (e.g., 'audio/mpeg', 'image/png', 'video/mp4') is recommended for accuracy, but the tool will attempt to guess it from the file extension if not provided."
    )
    args_schema: type[BaseModel] = MultimodalInput

    def _run(self, file_path: str, user_prompt: str, mime_type: Optional[str] = None) -> str:
        global llm  # A globális LLM példány használata
        if not llm:
            return "Error: LLM (ChatGoogleGenerativeAI instance) is not available."
        if not Path(file_path).exists():
            return f"Error: File not found at path: {file_path}"
        if not user_prompt:
            return "Error: 'user_prompt' (instruction) must be provided."

        content_parts = [{"type": "text", "text": user_prompt}]

        try:
            with open(file_path, "rb") as f:
                encoded_content = base64.b64encode(f.read()).decode("utf-8")

            actual_mime_type = mime_type
            if not actual_mime_type:
                extension_to_mime = {
                    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac", ".ogg": "audio/ogg",
                    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
                    ".mp4": "video/mp4", ".webm": "video/webm",
                }
                file_extension = Path(file_path).suffix.lower()
                actual_mime_type = extension_to_mime.get(file_extension)
                if not actual_mime_type:
                    return f"Error: Could not determine MIME type for file extension '{file_extension}'. Please provide 'mime_type'."

            print(f"  Determined MIME type: {actual_mime_type} for file: {file_path}")

            # Formázás a geminiflashapi_langchain.pdf alapján
            # Kép esetén (6. oldal): {"type": "image_url", "image_url": f"data:{actual_mime_type};base64,{encoded_content}"}
            # Audió esetén (8. oldal): {"type": "media", "data": encoded_content, "mime_type": actual_mime_type}
            # Videó esetén (10. oldal): {"type": "media", "data": encoded_content, "mime_type": actual_mime_type}

            # LangChain HumanMessage struktúrája a Google GenAI-hoz (általánosabb)
            # A `parts` listában a `blob` (mime_type, data) és `text` elemeket várja a Google API.
            # Langchain ezt lekezeli, ha a `content` egy lista, ahol a dict-eknek van pl. "type": "image_url" vagy "type": "text"
            # A PDF-ben lévő HumanMessage(content=[{"type":"text", ...}, {"type":"image_url", ...}]) vagy
            # HumanMessage(content=[{"type":"text", ...}, {"type":"media", ...}]) formátumokat kell követni.

            if actual_mime_type.startswith("image/"):
                content_parts.append({
                    "type": "image_url",  # Langchain konvenció képekre
                    "image_url": f"data:{actual_mime_type};base64,{encoded_content}"
                })
            elif actual_mime_type.startswith("audio/") or actual_mime_type.startswith("video/"):
                # A PDF 8. és 10. oldala alapján a "media" type és "data" + "mime_type" kulcsok kellenek.
                # Viszont a LangChain Google GenAI integráció lehet, hogy egyszerűbben kezeli, ha csak a base64 stringet adjuk meg
                # a megfelelő message content type-pal.
                # A Google API várja a mime_type-ot és a base64 adatot a blob részeként.
                # A legtisztább, ha a Langchain dokumentációt követjük a `ChatGoogleGenerativeAI` multimodális inputjaira.
                # A `geminiflashapi_langchain.pdf` 8. oldala (Audio Input) ezt a struktúrát használja:
                # HumanMessage(content=[ {"type": "text", "text": "Transcribe the audio."},
                #                        {"type": "media", "data": encoded_audio, "mime_type": audio_mime_type} ])
                # Ez tűnik a helyesnek.
                content_parts.append({
                    "type": "media",  # Ez egy általánosabb "blob" típusra utalhat a Google API-ban
                    "data": encoded_content,
                    "mime_type": actual_mime_type
                })
            else:
                return f"Error: Unsupported MIME type '{actual_mime_type}' for multimodal processing."

            message = HumanMessage(content=content_parts)

            print(f"  Invoking LLM (multimodal_file_processor) with user_prompt: '{user_prompt}' for file: {file_path}")
            ai_response = llm.invoke(
                [message])  # Itt a multimodális képességű llm-et hívjuk, nem tool-t akarunk vele hívatni.

            if isinstance(ai_response.content, str):
                return ai_response.content
            else:
                # Az LLM válasza lehet komplexebb, ha pl. képet is generálna, de mi szöveget várunk.
                # Próbáljuk meg összefűzni, ha lista stringekből.
                if isinstance(ai_response.content, list):
                    return " ".join(str(c) for c in ai_response.content)
                print(f"Warning: Multimodal processing returned non-string/list content: {type(ai_response.content)}")
                return str(ai_response.content)

        except FileNotFoundError:
            return f"Error: File not found at path: {file_path}"
        except Exception as e:
            import traceback
            return f"Error during multimodal processing for file {file_path}: {str(e)}\n{traceback.format_exc()}"

class ChessAnalysisInput(BaseModel):
    fen: str = Field(description="The Forsyth-Edwards Notation (FEN) string of the chess position.")
    # search_depth_ ply: Optional[int] = Field(default=1, description="How many plies (half-moves) deep to search for a checkmate. Max 2-3 for simple analysis without a full engine.") # Egyszerűsítésként ezt most nem használjuk

class ChessAnalysisTool(BaseTool):
    name: str = "chess_position_analyzer"
    description: str = (
        "Analyzes a chess position given in FEN notation using the Stockfish chess engine to find the best move "
        "or a winning move for the current player. Provide the FEN string. "
        "The tool will return the best move in algebraic notation if found, or an analysis."
    )
    args_schema: type[BaseModel] = ChessAnalysisInput # Feltételezve, hogy a ChessAnalysisInput már létezik és csak 'fen'-t vár

    # ÚJ: Add meg a Stockfish elérési útját itt, vagy tedd konfigurálhatóvá
    # Ezt környezeti változóból, konfigurációs fájlból vagy a tool inicializálásakor is megadhatnád.
    # Egyelőre beégethetjük, de később érdemes rugalmasabbá tenni.
    STOCKFISH_PATH: Optional[str] = "C:/Users/zozo/PycharmProjects/agentcourse/stockfish/stockfish-windows-x86-64-avx2.exe"
                                                                        # Linux/macOS: "/usr/games/stockfish" vagy "/opt/stockfish/stockfish" stb.

    def _run(self, fen: str) -> str:
        try:
            import chess
            import chess.engine
        except ImportError:
            return "Error: The 'python-chess' library is not installed. Please install it by running 'pip install python-chess'."

        board = chess.Board(fen)

        if board.is_game_over():
            # ... (a játék vége állapotok kezelése változatlanul maradhat)
            if board.is_checkmate():
                return f"Position is already checkmate. Result: {board.result()}"
            elif board.is_stalemate():
                return f"Position is stalemate. Result: {board.result()}"
            elif board.is_insufficient_material():
                return "Position is a draw due to insufficient material."
            else:
                return f"Game is over. Result: {board.result()}"

        if not self.STOCKFISH_PATH or not Path(self.STOCKFISH_PATH).exists(): # Path importálva kell legyen
            error_msg = (
                f"Error: Stockfish engine not found at the specified path: {self.STOCKFISH_PATH}. "
                "Please ensure Stockfish is installed and the STOCKFISH_PATH variable in ChessAnalysisTool is set correctly. "
                "Falling back to simple mate check."
            )
            print(error_msg) # Logoljuk a hibát
            # Fallback: Egyszerű mattkeresés, ha Stockfish nem elérhető
            for move in board.legal_moves:
                board.push(move)
                if board.is_checkmate():
                    board.pop()
                    return f"Found checkmate (simple search): {board.san(move)}"
                board.pop()
            return f"{error_msg} No immediate checkmate found with simple search. FEN: {fen}"


        try:
            # Gondolkodási idő a Stockfish számára (másodpercben)
            # GAIA feladatokhoz általában rövid idő is elég lehet egy erős lépéshez.
            # A "garantált nyerés" valószínűleg nem igényel extrém mély analízist.
            analysis_time_limit = 20.0 # 2 másodperc

            print(f"  ChessAnalysisTool: Analyzing FEN '{fen}' with Stockfish (time limit: {analysis_time_limit}s). Path: {self.STOCKFISH_PATH}")

            # A context manager (with ... as engine) biztosítja, hogy az engine leálljon a végén
            with chess.engine.SimpleEngine.popen_uci(self.STOCKFISH_PATH) as engine:
                # Beállíthatunk erősséget vagy más UCI opciókat, ha szükséges
                # engine.configure({"UCI_Elo": 1500}) # Példa az erősség korlátozására

                result = engine.play(board, chess.engine.Limit(time=analysis_time_limit))

                if result.move:
                    best_move_san = board.san(result.move)
                    # Esetleg a Stockfish értékelését is hozzáadhatjuk
                    # info = engine.analyse(board, chess.engine.Limit(time=0.1), multipv=1) # Gyors értékelés
                    # score = info[0].get('score')
                    # if score:
                    #     return f"Best move found: {best_move_san} (Score: {score})"
                    return f"FINAL_CHESS_MOVE: {best_move_san}"
                else:
                    return "Stockfish analysis did not return a best move within the time limit."

        except chess.engine.EngineTerminatedError:
            return "Error: Stockfish engine terminated unexpectedly."
        except chess.engine.EngineError as e:
            return f"Error: Stockfish engine error: {str(e)}"
        except FileNotFoundError: # Dupla ellenőrzés, bár a self.STOCKFISH_PATH ellenőrzés már megvolt
             return (
                f"Error: Stockfish executable not found at path: {self.STOCKFISH_PATH}. "
                "Please ensure Stockfish is installed and the path is correct."
            )
        except Exception as e:
            import traceback
            return f"Error during Stockfish chess analysis for FEN '{fen}': {str(e)}\n{traceback.format_exc()}"


class FenGeneratorInput(BaseModel):
    image_file_path: str = Field(description="The absolute path to the downloaded chess diagram image file.")
    # Opcionálisan itt is átadhatnánk a num_tries stb. paramétereket, ha az LLM-re bíznánk a finomhangolást


class ExternalFenGeneratorTool(BaseTool):
    name: str = "external_image_to_fen_generator"
    description: str = (
        "Generates a FEN string from a chess diagram image using an external, specialized script. "
        "Use this after downloading the image file. Provide the 'image_file_path'. "
        "The FEN string returned by this tool represents the piece placement but might need adjustment for whose turn it is."
    )
    args_schema: type[BaseModel] = FenGeneratorInput

    # === KONFIGURÁCIÓ ===
    # Ezt a két értéket pontosan be kell állítanod a te rendszerednek megfelelően!
    PICTOCODE_PYTHON_PATH: str = "C:/chromedriver/anaconda3/envs/pictocode/python.exe"  # VAGY /home/user/anaconda3/envs/pictocode/bin/python
    FEN_GENERATOR_SCRIPT_PATH: str = "C:/Users/zozo/PycharmProjects/Chess_diagram_to_FEN/run_fen_generator.py"  # Ahol a run_fen_generator.py fájlod van

    def _run(self, image_file_path: str) -> str:
        if not Path(self.PICTOCODE_PYTHON_PATH).exists():
            return f"Error: Python executable for 'pictocode' environment not found at {self.PICTOCODE_PYTHON_PATH}"
        if not Path(self.FEN_GENERATOR_SCRIPT_PATH).exists():
            return f"Error: FEN generator script (run_fen_generator.py) not found at {self.FEN_GENERATOR_SCRIPT_PATH}"

        abs_image_path = str(Path(image_file_path).resolve())  # Biztos, ami biztos, abszolút elérési út

        command = [
            self.PICTOCODE_PYTHON_PATH,
            self.FEN_GENERATOR_SCRIPT_PATH,
            "--image_path",
            abs_image_path
            # Itt adhatnánk át további argumentumokat a wrapper scriptnek, ha szükséges
            # "--num_tries", "15"
        ]

        print(f"  {self.name}: Executing command: {' '.join(command)}")

        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=60,  # Adjunk neki elég időt, a FEN generálás lehet lassú
                check=False  # Kézzel ellenőrizzük a visszatérési kódot és a stderr-t
            )

            if process.returncode != 0:
                return f"Error running FEN generator script. Return code: {process.returncode}. Stderr: {process.stderr.strip()}"

            fen_output = process.stdout.strip()
            if not fen_output:
                return f"Error: FEN generator script produced no output. Stderr: {process.stderr.strip()}"

            # Egyszerű ellenőrzés, hogy FEN-szerű-e (legalább egy '/' van benne)
            if "/" not in fen_output:
                return f"Error: Output from FEN generator does not look like a FEN string: '{fen_output}'. Stderr: {process.stderr.strip()}"

            print(f"  {self.name}: Successfully generated FEN (raw): {fen_output}")
            return fen_output

        except subprocess.TimeoutExpired:
            return f"Error: FEN generator script timed out after 60 seconds."
        except Exception as e:
            import traceback
            return f"An unexpected error occurred while running FEN generator: {str(e)}\n{traceback.format_exc()}"

class ModifyFenInput(BaseModel):
    fen: str = Field(description="The FEN string to modify.")
    player_to_move: str = Field(description="The player to set as active, 'w' for white or 'b' for black.")

class ModifyFenTool(BaseTool):
    name: str = "fen_active_player_modifier"
    description: str= "Modifies the 'player to move' part of a FEN string."
    args_schema: type[BaseModel] = ModifyFenInput

    def _run(self, fen: str, player_to_move: str) -> str:
        parts = fen.split(' ')
        if len(parts) >= 2:
            if player_to_move in ['w', 'b']:
                parts[1] = player_to_move
                return " ".join(parts)
            else:
                return f"Error: Invalid player_to_move '{player_to_move}'. Use 'w' or 'b'."
        return f"Error: FEN string '{fen}' does not have enough parts to modify player to move."

class YouTubeVideoAnalysisInput(BaseModel):
    video_url: str = Field(description="The URL of the YouTube video to analyze.")
    user_prompt: str = Field(description="The specific question or instruction for the LLM on what to do with the video content (e.g., 'Summarize this video', 'What is said about X?').")

class YouTubeVideoAnalysisTool(BaseTool):
    name: str = "youtube_video_analyzer"
    description: str = (
        "Analyzes the content of a YouTube video given its URL and a user prompt/question. "
        "Use this tool when a question refers to a YouTube video URL and requires understanding its content (e.g., summary, specific information mentioned). "
        "This tool attempts to process the video content directly using Google's generative AI capabilities. "
        "Do NOT use this for downloading general files or files attached via task_id; use 'file_downloader' for that."
    )
    args_schema: type[BaseModel] = YouTubeVideoAnalysisInput
    model_name: str = "gemini-2.5-flash-preview-05-20"  # Vagy a kívánt modell

    def _run(self, video_url: str, user_prompt: str) -> str:
        # --- 1. LÉPÉS: SDK Konfigurálása és Modell Példányosítása ---
        model_instance_for_youtube_tool = None  # Előre deklaráljuk, hogy a függvény végén is elérhető legyen (ha szükséges lenne)
        try:
            api_key_to_use = os.getenv("GOOGLE_API_KEY")
            if not api_key_to_use:
                # A GEMINI_API_KEY fallbackot kikommentáltad, ez rendben van, ha a GOOGLE_API_KEY az elsődleges.
                return "Error: GOOGLE_API_KEY not found for YouTube Analysis."

            # Helyi import, hogy biztosan a megfelelő SDK-t használjuk és konfiguráljuk
            import google.generativeai as local_google_genai_sdk

            local_google_genai_sdk.configure(api_key=api_key_to_use)

            # A self.model_name attribútumot használjuk, amit az osztály példányosításakor kap meg a tool
            model_instance_for_youtube_tool = local_google_genai_sdk.GenerativeModel(
                model_name=self.model_name  # Itt a self.model_name helyes
            )
            print(f"  YouTubeVideoAnalysisTool: Successfully configured and instantiated model '{self.model_name}'.")

        except Exception as e:
            # Fontos a részletes hibakiírás, hogy könnyebb legyen debuggolni
            import traceback
            error_msg = f"Error during google.generativeai setup in YouTubeTool: {str(e)}\n{traceback.format_exc()}"
            print(f"  YouTubeVideoAnalysisTool: {error_msg}")
            return error_msg  # Visszaadjuk a hibaüzenetet

        # --- 2. LÉPÉS: URL Validálása ---
        # Ez a GAIA specifikus URL ellenőrzés, tartsd meg, ahogy van, esetleg bővítsd, ha több mintát látsz.
        # A logodban https://www.youtube.com/watch?v=L1vXCYZAYYM szerepelt.
        valid_gaia_youtube_prefixes = [
            "youtube.com/watch?v=",
            "youtu.be/",
            "https://www.youtube.com/watch?v=...",  # Korábbi példák alapján
            "https://youtu.be/...",  # Korábbi példák alapján
            "https://www.youtube.com/watch?v=L1vXCYZAYYM"  # A mostani teszt alapján
        ]
        if not any(prefix in video_url for prefix in valid_gaia_youtube_prefixes):
            return f"Error: Invalid YouTube video URL provided for GAIA benchmark: {video_url}."

        # --- 3. LÉPÉS: API Hívás Előkészítése és Végrehajtása ---
        print(f"  YouTubeVideoAnalysisTool: Analyzing video URL '{video_url}'")
        print(f"  YouTubeVideoAnalysisTool: User prompt: '{user_prompt}'")
        # Az effective_model_name itt már nem szükséges külön, mert a model_instance_for_youtube_tool már a self.model_name alapján jött létre.
        print(f"  YouTubeVideoAnalysisTool: Using model '{self.model_name}' (from initialized instance)")

        try:
            # Az import google.generativeai.types itt is rendben van, ha a local_google_genai_sdk import sikeres volt.
            from google.generativeai.types import GenerationConfig as SDKGenerationConfig

            content_message_dict = {
                "role": "user",
                "parts": [
                    {"file_data": {"mime_type": "video/youtube", "file_uri": video_url}},
                    {"text": user_prompt}
                ]
            }
            contents_for_api = [content_message_dict]

            generation_config_settings = SDKGenerationConfig(
                response_mime_type="text/plain",
                temperature=0.2
            )

            # Itt már a korábban létrehozott 'model_instance_for_youtube_tool'-t használjuk!
            # A te kódodban a második `model_instance = genai.GenerativeModel(...)` sort törölted, ami helyes,
            # mert a `model_instance_for_youtube_tool`-t kell itt használni.
            # (Feltételezem, hogy a kódban, amit futtatsz, a `model_instance` a második try blokkban
            # valójában `model_instance_for_youtube_tool`-ra hivatkozik.)

            print(
                f"  YouTubeVideoAnalysisTool: Using GenAI model instance: {model_instance_for_youtube_tool.model_name}")  # Ellenőrizzük a nevét

            response = model_instance_for_youtube_tool.generate_content(  # Itt használjuk a helyes példányt
                contents=contents_for_api,
                generation_config=generation_config_settings,
                request_options={"timeout": 300}  # Jó ötlet a timeout növelése videóknál
            )

            # --- 4. LÉPÉS: Válasz Feldolgozása ---
            # A válaszfeldolgozó logika jónak tűnik.
            if hasattr(response, 'text') and response.text:
                print("  YouTubeVideoAnalysisTool: Successfully received response (direct text).")
                return response.text
            elif response.candidates and len(response.candidates) > 0 and response.candidates[0].content and \
                    response.candidates[0].content.parts:
                full_response_text = ""
                for part_dict in response.candidates[0].content.parts:
                    if "text" in part_dict:
                        full_response_text += part_dict["text"]

                if full_response_text:
                    print("  YouTubeVideoAnalysisTool: Successfully received response (from parts).")
                    return full_response_text
                else:
                    reason = response.candidates[0].finish_reason if response.candidates and len(
                        response.candidates) > 0 else "Unknown"
                    print(
                        f"  YouTubeVideoAnalysisTool: Response parts did not contain text. Finish reason: {reason}. Parts: {response.candidates[0].content.parts if response.candidates and len(response.candidates) > 0 else 'N/A'}")
                    return f"Error: LLM response format for video was not directly parsable to text (finish reason: {reason})."
            else:
                block_reason_msg = ""
                if hasattr(response,
                           'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason_msg = f" Blocked due to: {response.prompt_feedback.block_reason}."
                # A teljes response objektum logolása nagy lehet, óvatosan vele.
                print(
                    f"  YouTubeVideoAnalysisTool: LLM response for video was empty or in unexpected format.{block_reason_msg}")  # Esetleg a response egy részét logolhatod: Raw response parts: {str(response)[:200]}
                return f"Error: LLM response for video was empty or in an unexpected format.{block_reason_msg}"

        except ImportError as e_import:
            import traceback
            error_msg = f"Error: Failed to import SDK types for YouTube analysis. Detail: {str(e_import)}\n{traceback.format_exc()}"
            print(f"  YouTubeVideoAnalysisTool: {error_msg}")
            return error_msg
        except Exception as e_api_call:
            import traceback
            error_msg = f"Error during YouTube video analysis API call for URL {video_url}: {str(e_api_call)}\n{traceback.format_exc()}"
            print(f"  YouTubeVideoAnalysisTool: {error_msg}")
            return error_msg

"""
class GoogleNativeSearchInput(BaseModel):
    query: str = Field(description="The search query to find information using Google Search.")


class GoogleNativeSearchTool(BaseTool):
    name: str = "google_native_search"
    description: str = (
        "Performs a web search using Google's native search capabilities integrated with the Gemini model. "
        "Use this for general information seeking, finding facts, or when you need up-to-date information. "
        "The model will decide if a search is needed and use this tool to ground its response. "
        "Provide a clear and concise search query."
    )
    args_schema: type[BaseModel] = GoogleNativeSearchInput
    # A modell nevét az agent fő LLM-jéből vehetnénk, vagy itt fixálhatjuk.
    # Legyen konzisztens azzal, amit az agent többi részén használunk, ha lehetséges,
    # de a Google Search toolhoz lehet, hogy specifikus modellverzió ajánlott.
    # A mintakód 'gemini-2.0-flash'-t használ, mi 'gemini-2.5-flash-preview-05-20'-at.
    # Ennek működnie kell.
    search_model_name: str = "gemini-2.5-flash-preview-05-20"  # Vagy "gemini-2.0-flash", ha a 2.5 nem működne jól vele

    def _run(self, query: str) -> str:
        try:
            api_key_to_use = os.getenv("GOOGLE_API_KEY")
            if not api_key_to_use:
                api_key_to_use = os.getenv("GEMINI_API_KEY")
                if not api_key_to_use:
                    return "Error: GOOGLE_API_KEY (or GEMINI_API_KEY) not found for GoogleNativeSearchTool."

            genai.configure(api_key=api_key_to_use)
        except Exception as e:
            return f"Error during google.generativeai configuration for GoogleNativeSearchTool: {e}"

        print(f"  GoogleNativeSearchTool: Performing search for query: '{query}'")
        print(f"  GoogleNativeSearchTool: Using search model: '{self.search_model_name}'")

        try:
            # Importáljuk a szükséges típusokat a google.genai.types-ból
            from google.generativeai.types import Tool as SDKTool
            from google.generativeai.types import GoogleSearch as SDKGoogleSearch
            # A GenerateContentConfig-ot most szótárként fogjuk definiálni

            # Google Search eszköz létrehozása
            google_search_sdk_tool = SDKTool(google_search=SDKGoogleSearch())

            # Generálási konfiguráció SZÓTÁRKÉNT
            generation_config_dict = {
                "tools": [google_search_sdk_tool],
                # "response_modalities": ["TEXT"], # Opcionális, a default általában text
                "temperature": 0.1,  # Konzisztens válaszokért
                # Ha más GenerateContentConfig paramétereket is használni szeretnél, itt add hozzá őket
                # pl. "candidate_count": 1
            }

            model_instance = genai.GenerativeModel(
                model_name=self.search_model_name
            )

            print(f"  GoogleNativeSearchTool: Using GenAI model instance: {model_instance.model_name}")

            response = model_instance.generate_content(
                contents=query,
                generation_config=generation_config_dict,  # Itt a szótárat adjuk át
                request_options={"timeout": 120}
            )

            # Válaszfeldolgozás (változatlan marad)
            response_text_parts = []
            if response.candidates and len(response.candidates) > 0 and response.candidates[0].content and \
                    response.candidates[0].content.parts:
                for part_dict in response.candidates[0].content.parts:
                    if "text" in part_dict:
                        response_text_parts.append(part_dict["text"])

            final_response_text = "\n".join(response_text_parts).strip()

            if final_response_text:
                print(f"  GoogleNativeSearchTool: Successfully received search response.")
                return final_response_text
            else:
                reason = response.candidates[0].finish_reason if response.candidates and len(
                    response.candidates) > 0 else "Unknown"
                block_reason_msg = ""
                if hasattr(response,
                           'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason_msg = f" Blocked due to: {response.prompt_feedback.block_reason}."
                print(
                    f"  GoogleNativeSearchTool: No text found in search response parts. Finish reason: {reason}.{block_reason_msg}")
                return f"Error: Google Search did not return a text response (finish reason: {reason}).{block_reason_msg}"

        except ImportError:  # Ha az SDKTool vagy SDKGoogleSearch import nem sikerül
            return "Error: Failed to import required types (Tool, GoogleSearch) from 'google.genai.types'. SDK issue."
        except Exception as e:
            import traceback
            error_msg = f"Error during Google Native Search for query '{query}': {str(e)}\n{traceback.format_exc()}"
            print(f"  GoogleNativeSearchTool: {error_msg}")
            return error_msg
"""

def execute_external_google_search_script(query: str,
                                          script_path: str = "external_google_search_EXACT.py",
                                          # EZT A SZKRIPTET HÍVJUK
                                          model_id_for_script: str = "gemini-2.5-flash-preview-05-20") -> str:  # A modell, amit a szkript használ
    """
    Meghívja a külső Python szkriptet a Google keresés végrehajtásához.
    """
    print(f"  Executing external Google search script ('{script_path}') for query: '{query}'")
    try:
        process = subprocess.run(
            [sys.executable, script_path, query, model_id_for_script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False
        )

        if process.returncode != 0:
            error_output = process.stderr.strip() if process.stderr else "Unknown error in script."
            if process.stdout and "ERROR_" in process.stdout.upper():  # Figyeljünk a szkript ERROR_ prefixére
                error_output = process.stdout.strip()
            print(f"  Error running external search script (code {process.returncode}): {error_output}")
            return f"Error running external search script: {error_output}"

        output = process.stdout.strip()
        if output.upper().startswith("ERROR_"):  # Figyeljünk a szkript ERROR_ prefixére
            print(f"  Error from external search script: {output}")
            return f"Error from external search script: {output}"

        if not output:
            print("  Warning: External search script produced no output, but exited successfully.")
            return "External search script produced no output."  # Vagy egy specifikusabb üzenet

        return output

    except FileNotFoundError:
        return f"Error: Search script '{script_path}' not found."
    except subprocess.TimeoutExpired:
        return "Error: External search script timed out."
    except Exception as e:
        return f"Error calling external search script: {str(e)}"


def prepare_search_sub_graph(state: AgentState) -> AgentState:
    print("--- PREPARING STATE FOR SEARCH SUB-GRAPH ---")

    # A fő LLM által javasolt query-t a current_tool_input-ból vesszük,
    # ha a plan_next_step oda tette az {"action": "INITIATE_SEARCH_SUB_GRAPH", "initial_query": "..."} részeként.
    # Vagy ha a current_tool_input közvetlenül a query string. Ezt a plan_next_step-ben kell konzisztensen kezelni.
    # Tegyük fel, hogy a current_tool_input egy dict, amiben van "initial_query" kulcs.
    initial_query_from_main_llm = ""
    if isinstance(state.get("current_tool_input"), dict):
        initial_query_from_main_llm = state.get("current_tool_input", {}).get("initial_query", "")

    if not initial_query_from_main_llm:
        # Ha valamiért nincs query, adjunk egy általánosat, vagy jelezzünk hibát.
        # Ez nem szabadna, hogy előforduljon, ha a plan_next_step jól működik.
        print("  WARNING: No initial_query provided by main LLM for search sub-graph. Using original question.")
        initial_query_from_main_llm = state.get("original_question", "Missing original question for search.")

    return {
        **state,
        "search_sub_graph_active": True,
        "search_original_query_from_main_llm": initial_query_from_main_llm,
        "search_current_refined_query": None,  # Az első körben az LLM finomíthatja ezt
        "search_results_history": [],
        "search_analysis_history": [],
        "search_iteration_count": 0,
        "search_max_sub_iterations": state.get("search_max_sub_iterations", DEFAULT_MAX_SEARCH_SUB_ITERATIONS),
        "search_processing_status": "starting_search_sub_graph",
        "search_final_summary_for_main_agent": None,
        "current_tool_name": None,  # Töröljük az action nevet
        "current_tool_input": None  # Töröljük az action inputot
    }


def execute_native_search_node(state: AgentState) -> AgentState:
    print(f"--- SEARCH SUB-GRAPH (Iter: {state.get('search_iteration_count', 0)}): EXECUTING EXTERNAL GOOGLE SEARCH SCRIPT ---")

    query_to_execute = state.get("search_current_refined_query")
    if not query_to_execute:
        # Ha nincs finomított query (pl. az első körben a refiner nem adott, vagy hiba történt),
        # használjuk az eredetit, amit a fő LLM adott.
        query_to_execute = state.get("search_original_query_from_main_llm", "")
        if not query_to_execute:
            error_msg = "SEARCH_SUB_GRAPH_ERROR: No query available to execute search."
            print(f"  {error_msg}")
            # Ezt a hibát a search_results_history-be is beírhatnánk, hogy az LLM lássa.
            # Vagy közvetlenül a search_final_summary_for_main_agent-be és kilépünk.
            return {
                **state,
                "search_results_history": state.get("search_results_history", []) + [("NO_QUERY_PROVIDED", error_msg)],
                "search_processing_status": "search_execution_failed_no_query",
                "search_final_summary_for_main_agent": error_msg  # Kilépéshez
            }

    print(f"  Executing search with query: '{query_to_execute}'")

    # Itt hívjuk a te működő függvényedet.
    # Győződj meg róla, hogy a `perform_native_google_search` elérhető.
    # A modell ID-t is beállíthatod itt, vagy az agent state-ből veszed.
    #search_model_id = state.get("model_id_for_search_tool", "gemini-2.5-flash-preview-05-20")  # Példa
    search_output = execute_external_google_search_script(query=query_to_execute, model_id_for_script="gemini-2.5-flash-preview-05-20")

    print(f"  External Google Script FULL output:\n{search_output}\n--------------------")

    updated_search_history = state.get("search_results_history", []) + [(query_to_execute, search_output)]

    return {
        **state,
        "search_results_history": updated_search_history,
        "search_current_refined_query": None,  # Töröljük, hogy a következő körben újra generálódjon, ha kell
        "search_iteration_count": state.get("search_iteration_count", 0) + 1,
        "search_processing_status": "external_search_executed"
    }


def finalize_search_sub_graph(state: AgentState) -> AgentState:
    print(f"--- SEARCH SUB-GRAPH (Iter: {state.get('search_iteration_count', 0)}): FINALIZING ---")

    final_summary = state.get("search_final_summary_for_main_agent")
    current_status = state.get("search_processing_status")
    current_iter = state.get("search_iteration_count", 0)
    max_iter = state.get("search_max_sub_iterations", DEFAULT_MAX_SEARCH_SUB_ITERATIONS)

    if not final_summary:
        if current_status == "max_search_iterations_reached" or \
                (current_status not in ["final_search_summary_ready", "search_llm_indicated_failure"] and current_iter >= max_iter) or \
                current_status == "search_llm_indicated_failure" or \
           current_status == "search_execution_failed_no_query":
            final_summary = f"SEARCH_SUB_GRAPH_FAILED: Max iterations ({current_iter}/{max_iter}) reached. No conclusive summary generated."
        elif state.get("search_results_history"):
            # Ha nincs explicit összefoglaló, de volt keresés, próbáljuk meg az utolsó eredményt (vagy egy részét)
            # Ez lehet, hogy nem ideális, de jobb, mint a semmi.
            last_query, last_output = state['search_results_history'][-1]
            if "ERROR" in str(last_output).upper() or "FAILED" in str(last_output).upper():
                final_summary = f"SEARCH_SUB_GRAPH_ENDED: Last search attempt for '{last_query}' resulted in error: {str(last_output)[:200]}"
            else:
                final_summary = f"SEARCH_SUB_GRAPH_COMPLETED_NO_LLM_SUMMARY: Last search result for query '{last_query}' was: {str(last_output)[:500]}... (Full analysis by main agent might be needed)"
        else:
            final_summary = f"SEARCH_SUB_GRAPH_FAILED: No summary and no search history. Status: {current_status}"

    print(f"  Search Sub-Graph final summary/result for main agent: {final_summary[:300]}...")

    # Hozzáadjuk a fő agent intermediate_steps-éhez
    main_intermediate_steps = state.get('intermediate_steps', [])
    search_sub_graph_output_entry = ("search_sub_graph_result", final_summary)  # Egyértelmű jelzés
    updated_main_intermediate_steps = main_intermediate_steps + [search_sub_graph_output_entry]

    # Döntés arról, hogy a `final_answer` beállításra kerüljön-e a fő ágon
    # Általában a keresési al-ág csak információt szolgáltat, a fő LLM dönt a végső válaszról.
    # Tehát itt a final_answer-t nem bántjuk.

    main_error_message = state.get("error_message")
    if "FAILED" in final_summary.upper() or "ERROR" in final_summary.upper():
        if not main_error_message or "SEARCH_SUB_GRAPH" in final_summary:
            main_error_message = final_summary  # Felülírjuk, ha a keresés volt a hiba
    else:  # Ha a keresés sikeres volt (vagy legalábbis nem jelzett hibát), töröljük a keresési hibára utaló error_message-t, ha volt
        if main_error_message and "SEARCH_SUB_GRAPH" in main_error_message:
            main_error_message = None

    return {
        **state,
        "intermediate_steps": updated_main_intermediate_steps,
        "search_sub_graph_active": False,  # Nagyon fontos a kikapcsolás!
        "error_message": main_error_message, #state.get("error_message") if "FAILED" not in final_summary.upper() else final_summary,
        # Csak akkor írjuk felül, ha a keresés hibát jelzett
        # A keresési al-ág specifikus mezőit itt nullázhatjuk, ha akarjuk, de nem kötelező
        # "search_original_query_from_main_llm": None,
        # "search_current_refined_query": None,
        # "search_results_history": [],
        # "search_analysis_history": [],
        # "search_processing_status": "search_completed",
        # "search_iteration_count": 0,
        "search_final_summary_for_main_agent": None  # Ezt már feldolgoztuk
    }


def search_query_refiner_or_planner_llm(state: AgentState) -> AgentState:
    print(f"--- SEARCH SUB-GRAPH (Iter: {state.get('search_iteration_count', 0)}): QUERY REFINER/PLANNER LLM ---")

    if not state.get("search_sub_graph_active"):
        # Ez nem szabadna, hogy megtörténjen, ha a routerek jól működnek
        return {**state, "error_message": "Search sub-graph called when not active.",
                "search_processing_status": "sub_graph_inactive_error_at_llm_planner"}

    print(f"  DEBUG: Current search_results_history: {state.get('search_results_history')}")

    # Összeállítjuk a system promptot ennek az LLM hívásnak
    original_query = state.get("search_original_query_from_main_llm", "N/A")
    current_iter = state.get("search_iteration_count", 0)
    max_iter = state.get("search_max_sub_iterations", DEFAULT_MAX_SEARCH_SUB_ITERATIONS)

    # Előzmények formázása a prompt számára
    results_history_for_prompt = []
    MAX_SEARCH_HISTORY_IN_PROMPT = 2  # Mennyi korábbi keresést mutassunk

    # Az utolsó X elemet vesszük a search_results_history-ből
    recent_searches = state.get("search_results_history", [])[-MAX_SEARCH_HISTORY_IN_PROMPT:]

    total_attempts = len(state.get("search_results_history", []))

    for i, (q, r) in enumerate(recent_searches):
        # Az 'attempt_num' az abszolút sorszámot mutatja az összes próbálkozásból
        attempt_num = total_attempts - len(recent_searches) + 1 + i
        RESULT_TRUNCATION_LIMIT = 5000  # Növeljük meg ezt az értéket
        results_history_for_prompt.append(
            f"  Search Attempt {attempt_num}:\n"
            f"    Query: \"{q}\"\n"
            f"    Result: \"{str(r)[:RESULT_TRUNCATION_LIMIT]}... (truncated at {RESULT_TRUNCATION_LIMIT} chars if longer)\""
        )

    # Ha fordított időrendben akarjuk a promptban (legfrissebb legfelül):
    results_history_str = "\n---\n".join(
        reversed(results_history_for_prompt)) if results_history_for_prompt else "No recent search attempts to show."
    # Vagy ha normál időrendben (legrégebbi a legfrissebbek közül felül):
    # results_history_str = "\n---\n".join(results_history_for_prompt) if results_history_for_prompt else "No recent search attempts to show."

    analysis_history_str = "No analysis of previous search results has been performed yet in this sub-task."  # Frissített szöveg

    """
    for i, (sr, an) in enumerate(state.get("search_analysis_history", [])):
        analysis_history_str_parts.append(
            f"  Analysis {i + 1} on Search Result: \"{str(sr)[:500]}...\"\n  Analysis {i + 1} Text: \"{str(an)[:500]}...\"")
    analysis_history_str = "\n---\n".join(
        analysis_history_str_parts) if analysis_history_str_parts else "No analysis yet."
    """
    search_sub_graph_system_prompt = f"""You are a specialized AI assistant within a larger agent, tasked with resolving a search query.
Your goal is to find relevant information and provide a concise summary that helps answer the main agent's original question, or to refine the search query if needed.

Original query from main agent that initiated this search sub-task: "{original_query}"
Current iteration in this search sub-task: {current_iter + 1} / {max_iter}

Most Recent Search Attempts and Results (if any, max {MAX_SEARCH_HISTORY_IN_PROMPT} shown, most recent first):
---
{results_history_str}
---

Analysis History (your previous analyses of search results in this sub-graph):
---
{analysis_history_str}
---

Your task now:
1.  **Carefully analyze the `Original query from main agent` AND all entries in the `Search Results History`.**
2.  **Determine if the information needed to answer the `Original query from main agent` is ALREADY PRESENT in the `Search Results History`.**
    *   Pay close attention to all constraints in the original query (e.g., dates, specific conditions like "country that no longer exists").
    *   For example, if the original query asks for a person from a country that no longer exists, and a search result lists "East Germany", this is a relevant finding.

3.  **Decide your next step:**
    *   **PRIORITY 1 - OPTION B: Analyze Results & Provide Summary:** 
        IF the `Search Results History` **CONTAINS CLEAR AND SUFFICIENT INFORMATION** to directly answer or make significant progress on the `Original query from main agent`, THEN you MUST choose this option.
        Analyze the relevant search results and provide a concise summary. Your output MUST BE ONLY a JSON object string:
        `{{"action": "PROVIDE_FINAL_SEARCH_SUMMARY", "summary": "Your concise summary here, directly addressing the original query based on the search results. State the findings clearly."}}`
        **Example:** If a search result shows "Claus Peter Flor (East Germany)", and the original query asks for a person from a country that no longer exists, your summary should highlight this.

    *   **OPTION A: Refine Query & Search Again:** 
        IF AND ONLY IF Option B is not yet possible (i.e., the current information is clearly insufficient, a previous search explicitly failed with an error message, or the results are entirely irrelevant) AND you are within the iteration limit ({max_iter}), formulate a **NEW and DIFFERENT, more effective** search query. 
        If the previous query was good but just needs a small tweak, that's acceptable.
        **DO NOT submit the exact same query if it already produced some results unless those results were explicit errors.**
        Your output MUST BE ONLY a JSON object string:
        `{{"action": "SUBMIT_NEW_SEARCH_QUERY", "query": "your new, different, and refined search query here"}}`

    *   **OPTION C: Cannot Proceed / Max Iterations:** 
        If you have reached the maximum iterations ({max_iter}), OR if after several distinct search attempts you cannot find relevant information, OR if an unrecoverable error occurred, indicate failure. Your output MUST BE ONLY a JSON object string:
        `{{"action": "SEARCH_FAILED_OR_MAX_ITERATIONS", "reason": "Explain briefly (e.g., max iterations reached, no relevant results found for queries like 'X', 'Y', 'Z', or specific error encountered)."}}`

Provide ONLY the JSON object string for your chosen option. Do not add any other text, explanations, or conversational filler before or after the JSON."""

    # Az LLM hívása (ugyanazt az `llm` példányt használjuk, mint a fő agent)
    # A `tools` itt nem releváns, mert az LLM-nek JSON-t kell visszaadnia.
    messages_for_search_llm = [
        SystemMessage(content=search_sub_graph_system_prompt),
        HumanMessage(
            content="Based on the current state and history, what is your next step (provide as a JSON object)?")
    ]

    print("  Invoking LLM for search sub-graph planning/analysis...")
    try:
        # Itt nem használunk `bind_tools`-t, mert specifikus JSON választ várunk.
        ai_response_search_sub_graph: AIMessage = llm.invoke(messages_for_search_llm)
        response_content_str = ai_response_search_sub_graph.content

        if not isinstance(response_content_str, str):  # Biztosítjuk, hogy string legyen
            response_content_str = str(response_content_str)
            print(f"  Search Sub-Graph LLM raw response content (converted to string): '{response_content_str}'")

        else:
            print(f"  Search Sub-Graph LLM raw response content: '{response_content_str}'")

        # Próbáljuk meg a JSON-t kicsit robusztusabban parse-olni, hátha az LLM extra szöveget ad
        json_match = re.search(r"\{.*\}", response_content_str, re.DOTALL)
        if not json_match:
            print("  Search Sub-Graph LLM response did not contain a JSON-like structure.")
            return {**state, "search_processing_status": "search_llm_invalid_response_not_json",
                    "search_final_summary_for_main_agent": "Error: Search sub-agent LLM did not provide a JSON-like action."}

        json_string_to_parse = json_match.group(0)

        import json

        parsed_action = json.loads(json_string_to_parse)

        action_type = parsed_action.get("action")

        if action_type == "SUBMIT_NEW_SEARCH_QUERY":
            new_query = parsed_action.get("query")
            if not new_query or not isinstance(new_query, str) or not new_query.strip():
                return {**state, "search_processing_status": "search_llm_error_no_query_in_action", "search_final_summary_for_main_agent": "Error: LLM action SUBMIT_NEW_SEARCH_QUERY had missing or invalid query."}
            print(f"  Search Sub-Graph LLM decided to refine query to: '{new_query}'")
            return {
                **state,
                "search_current_refined_query": new_query.strip(),
                "search_processing_status": "awaiting_refined_search_execution"
            }
        elif action_type == "PROVIDE_FINAL_SEARCH_SUMMARY":
            summary = parsed_action.get("summary")
            if not summary or not isinstance(summary, str) or not summary.strip():
                return {**state,
                        "search_processing_status": "search_llm_error_no_summary_in_action",
                        "search_final_summary_for_main_agent": "Error: LLM action PROVIDE_FINAL_SEARCH_SUMMARY had missing or invalid summary."}
            print(f"  Search Sub-Graph LLM provided final summary: '{summary[:200]}...'")
            # Az elemzést is elmenthetjük, ha a prompt kérte volna, hogy a summary-t az analysis history-ba tegye.
            # Most közvetlenül a final_summary_for_main_agent-be tesszük.
            return {
                **state,
                "search_final_summary_for_main_agent": summary.strip(),
                "search_processing_status": "final_search_summary_ready"
            }
        elif action_type == "SEARCH_FAILED_OR_MAX_ITERATIONS":
            reason = parsed_action.get("reason", "No specific reason provided by search LLM.")
            print(f"  Search Sub-Graph LLM indicated failure/max_iterations: {reason}")
            return {
                **state,
                "search_final_summary_for_main_agent": f"Search Sub-Graph Failed: {reason}",
                "search_processing_status": "search_llm_indicated_failure"  # Ez jelzi a finalize-nak a kilépést
            }
        else:
            print(f"  Search Sub-Graph LLM returned unknown action in JSON: {action_type}")
            return {
                **state,
                "search_processing_status": "search_llm_unknown_action_in_json",
                "search_final_summary_for_main_agent": f"Error: Search sub-agent LLM provided an unknown action: {action_type}"
            }

    except json.JSONDecodeError:
        print(f"  Search Sub-Graph LLM response was not valid JSON: '{response_content_str}'")
        return {**state, "search_processing_status": "search_llm_json_decode_error",
                "search_final_summary_for_main_agent": "Error: Search sub-agent LLM response was not valid JSON."}

    except Exception as e:
        print(f"  Error during Search Sub-Graph LLM call or processsing: {e}")
        #import traceback

        traceback.print_exc()
        return {**state, "search_processing_status": "search_llm_exception",
                "search_final_summary_for_main_agent": f"Error in search sub-agent LLM: {str(e)[:200]}"}


def decide_next_search_step(state: AgentState) -> str:
    print(f"--- SEARCH SUB-GRAPH (Iter: {state.get('search_iteration_count', -1)}): DECIDING NEXT STEP ---")

    current_search_status = state.get("search_processing_status")
    current_search_iter = state.get("search_iteration_count", 0)
    max_search_iter_for_sub = state.get("search_max_sub_iterations", DEFAULT_MAX_SEARCH_SUB_ITERATIONS)

    # A kritikus hibák vagy kész állapotok listája
    finalize_statuses = [
        "final_search_summary_ready", "search_llm_indicated_failure",
        "search_llm_invalid_response_format", "search_llm_json_decode_error",
        "search_llm_unknown_action", "search_llm_error_no_query_in_action",
        "search_llm_error_no_summary_in_action", "search_execution_failed_no_query",
        "sub_graph_inactive_error_at_llm_planner"  # Hozzáadva a plannerből jövő hiba
    ]
    if current_search_status in finalize_statuses:
        print(
            f"  Search Sub-Graph: Final state or critical error ('{current_search_status}'). Routing to finalize_search_sub_graph.")
        return "finalize_search_sub_graph"  # Közvetlenül a node neve

    if current_search_iter >= max_search_iter_for_sub:
        print(
            f"  Search Sub-Graph: Max iterations ({max_search_iter_for_sub}) reached. Routing to finalize_search_sub_graph.")
        # A finalize node majd beállítja a search_final_summary_for_main_agent-et.
        # Itt beállíthatjuk a státuszt, hogy a finalize tudja, miért léptünk ki:
        # state["search_processing_status"] = "max_search_iterations_reached" # Ezt a finalize is megteheti, vagy itt expliciten
        # De jobb, ha a finalize node kezeli ezt a logikát a current_iter alapján.
        return "finalize_search_sub_graph"  # Közvetlenül a node neve

    # Ha az LLM új keresési lekérdezést adott (és még nem értük el a max iterációt)
    if current_search_status == "awaiting_refined_search_execution":
        print("  Search Sub-Graph: New query generated by LLM. Routing to execute_native_search_node.")
        return "execute_native_search_node"  # Közvetlenül a node neve

    # Ha a keresés lefutott, vagy ez az első lépés, vagy az LLM nem adott egyértelmű utasítást,
    # akkor újra az LLM-hez megyünk tervezni/elemezni.
    # Ide tartozik a "starting_search_sub_graph" és a "native_search_executed" státusz is.
    print(f"  Search Sub-Graph: Defaulting to LLM for planning/analysis. Status: '{current_search_status}' (e.g., starting_search_sub_graph, native_search_executed).")
    return "search_query_refiner_or_planner_llm"

"""
class LangchainGoogleSearchInput(BaseModel):
    query: str = Field(description="The search query to find information using Google Search.")


class LangchainGoogleSearchTool(BaseTool):
    name: str = "google_search"  # Maradhat ez a név, vagy lehet "langchain_google_search"
    description: str = (
        "A wrapper around Google Search. "
        "Useful for when you need to answer questions about current events, "
        "find facts, or get up-to-date information. Input should be a search query."
    )
    args_schema: type[BaseModel] = LangchainGoogleSearchInput

    # A wrapper inicializálása a _run metóduson belül történik, hogy biztosan friss legyen,
    # és hogy a környezeti változók betöltődjenek előtte.
    # Vagy az __init__-ben is lehetne, ha a környezeti változók ott már biztosan elérhetők.

    api_wrapper: GoogleSearchAPIWrapper = None  # Inicializáljuk None-nal

    def __init__(self, **kwargs):  # Hozzáadjuk az __init__ metódust
        super().__init__(**kwargs)
        try:
            # Itt ellenőrizzük a környezeti változókat, de a tényleges példányosítás a _run-ban lesz,
            # hogy minden híváskor friss legyen, vagy ha a kulcsok futás közben változnának (ritka).
            # De jobb itt példányosítani, hogy a hibát hamarabb elkapjuk.
            if not os.getenv("GOOGLE_API_KEY"):
                print("Warning: GOOGLE_API_KEY not found in environment for LangchainGoogleSearchTool.")
            if not os.getenv("GOOGLE_CSE_ID"):
                print("Warning: GOOGLE_CSE_ID not found in environment for LangchainGoogleSearchTool.")

            # Itt hozzuk létre a wrapper példányt.
            # A k=3 azt jelenti, hogy alapból 3 eredményt kérünk le. Ezt módosíthatod.
            self.api_wrapper = GoogleSearchAPIWrapper(k=2)
            print("LangchainGoogleSearchTool: GoogleSearchAPIWrapper initialized.")
        except Exception as e:
            print(f"LangchainGoogleSearchTool: Error initializing GoogleSearchAPIWrapper: {e}")
            # Ha itt hiba van, a tool nem lesz használható. A _run-ban is ellenőrizni kell.
            self.api_wrapper = None

    def _run(self, query: str) -> str:
        print(f"  LangchainGoogleSearchTool: Performing search for query: '{query}'")

        if self.api_wrapper is None:
            # Újrapróbálkozás az inicializálással, ha az __init__-ben nem sikerült,
            # vagy ha expliciten itt akarjuk tartani a példányosítást.
            # De jobb, ha az __init__-ben már megtörténik.
            # Ha az __init__-ben sikertelen volt, itt is az lesz valószínűleg.
            try:
                if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_CSE_ID"):
                    return "Error: GOOGLE_API_KEY or GOOGLE_CSE_ID not set for LangchainGoogleSearchTool."
                self.api_wrapper = GoogleSearchAPIWrapper(k=3)
                print("  LangchainGoogleSearchTool: GoogleSearchAPIWrapper re-initialized in _run.")
            except Exception as e:
                return f"Error initializing GoogleSearchAPIWrapper in _run: {e}"

        try:
            # A GoogleSearchAPIWrapper 'run' metódusa egy stringet ad vissza a keresési eredményekkel.
            # Ha a 'results' metódust használnánk, az egy lista dictionary-t adna vissza
            # (snippet, title, link), amit az LLM-nek nehezebb lehet feldolgozni közvetlenül.
            # A 'run' egyszerűbb, szöveges kimenetet ad.
            search_results_str = self.api_wrapper.run(query)

            if not search_results_str or search_results_str.strip() == "":
                print(f"  LangchainGoogleSearchTool: No results found for query '{query}'.")
                return f"No results found for query: {query}"

            print(
                f"  LangchainGoogleSearchTool: Search results obtained (first 300 chars): {search_results_str[:300]}...")
            return search_results_str

        except Exception as e:
            import traceback
            error_msg = f"Error during Langchain Google Search for query '{query}': {str(e)}\n{traceback.format_exc()}"
            print(f"  LangchainGoogleSearchTool: {error_msg}")
            return error_msg
"""

# === Eszközök Listájának Összeállítása ===
tools = [FileDownloaderTool()]
#if os.getenv("TAVILY_API_KEY"):  # Ellenőrizzük, hogy tényleg van-e kulcs
#    tools.append(TavilyWebSearchTool())
#    print("TavilyWebSearchTool added to tools list.")
#else:
#    print("TAVILY_API_KEY not found, TavilyWebSearchTool NOT added to tools list.")


"""
if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID"): # Csak akkor adjuk hozzá, ha a kulcsok megvannak
    tools.append(LangchainGoogleSearchTool())
    print("LangchainGoogleSearchTool added to tools list.")
else:
    print("LangchainGoogleSearchTool NOT added: GOOGLE_API_KEY or GOOGLE_CSE_ID missing.")
    """

tools.append(PythonInterpreterTool())
print("PythonInterpreterTool added to tools list.")

tools.append(MultimodalProcessingTool())
print("MultimodalProcessingTool added to tools list.")

tools.append(ChessAnalysisTool())
print("ChessAnalysisTool added to tools list.")

tools.append(ExternalFenGeneratorTool())
print("ExternalFenGeneratorTool added to tools list.")

tools.append(ModifyFenTool())
print("ModifyFenTool added to tools list.")

tools.append(YouTubeVideoAnalysisTool()) # <<< ÚJ ESZKÖZ HOZZÁADVA ITT
print("YouTubeVideoAnalysisTool added to tools list.")

"""
tools.append(GoogleNativeSearchTool())
print("GoogleNativeSearchTool added to tools list.")
"""

# === Agent Csomópontok ===

def initialize_agent(state: AgentState) -> AgentState:
    """Az LLM-et hívja meg, hogy megtervezze a következő lépést vagy a választ."""
    print("--- AGENT INITIALIZING ---")
    # Nem töröljük az original_question és task_id-t, mert azok a bemenetből jönnek
    current_question = state.get('original_question', "N/A")
    current_task_id = state.get('task_id', "N/A")

    return AgentState(
        original_question=current_question,
        task_id=current_task_id,
        file_path=None,
        plan=None,
        intermediate_steps=[],
        iterations=0,
        final_answer=None,
        error_message=None,
        current_tool_name=None,
        current_tool_input=None
    )

def prepare_excel_processing_state(state: AgentState) -> AgentState:
    print("--- PREPARING STATE FOR EXCEL SUB-GRAPH ---")
    # Az excel_max_sub_iterations értékét itt vagy a híváskor kell beállítani
    MAX_EXCEL_ITERATIONS_FOR_SUB_AGENT = 7 # Például
    return {
        **state,
        "excel_processing_active": True,
        "excel_original_question": state['original_question'], # A fő kérdés
        "excel_file_to_process": state['file_path'],       # A letöltött Excel fájl
        "excel_iteration_count": 0,
        "excel_max_sub_iterations": MAX_EXCEL_ITERATIONS_FOR_SUB_AGENT,
        "excel_code_execution_history": [],
        "excel_processing_status": "starting_excel_processing",
        "current_tool_name": None, # Töröljük a "virtuális" tool nevet, ha volt
        "current_tool_input": None
    }

def excel_llm_planner_coder(state: AgentState) -> AgentState:
    print(f"--- EXCEL SUB-AGENT (Iter: {state.get('excel_iteration_count', 0)}): PLANNING & CODE GENERATION ---")
    if not state.get('excel_file_to_process'):
        print("  EXCEL SUB-AGENT: ERROR - No Excel file path found in state.")
        return {**state, "excel_final_result_from_sub_agent": "EXCEL_PROCESSING_FAILED: Critical - No file path.", "excel_processing_active": False, "excel_processing_status": "critical_error_no_file"}

    excel_llm_instance_for_this_node = None
    MODEL_FOR_EXCEL_BRANCH = "gemini-2.5-flash-preview-05-20"
    # Vagy ha a teszt a gemini-1.5-flash-latest-tel ment, akkor azt használd:
    # MODEL_FOR_EXCEL_BRANCH = "gemini-1.5-flash-latest"

    # Szerezzük be a PythonInterpreterTool-t a globális 'tools' listából
    # Feltételezzük, hogy a 'tools' lista globálisan elérhető és tartalmazza a példányosított eszközöket.
    python_tool_instance_for_excel = next((t for t in tools if t.name == "python_interpreter"), None)

    if not python_tool_instance_for_excel:
        print("  EXCEL SUB-AGENT: ERROR - PythonInterpreterTool instance not found in global 'tools' list.")
        return {
            **state,
            "excel_final_result_from_sub_agent": "EXCEL_PROCESSING_FAILED: PythonInterpreterTool not configured for Excel branch.",
            "excel_processing_active": False,
            "excel_processing_status": "critical_error_no_python_tool_instance"
        }

    try:
        # Itt hozzuk létre az új LLM példányt, és CSAK a python_interpreter_tool-t bind-oljuk hozzá.
        # Az API kulcsot a környezeti változóból fogja venni (ahogy a sikeres tesztben).
        excel_llm_instance_for_this_node = ChatGoogleGenerativeAI(
            model=MODEL_FOR_EXCEL_BRANCH,
            temperature=0.1  # Kódgeneráláshoz alacsonyabb
        ).bind_tools([python_tool_instance_for_excel])  # Csak ezt az egy eszközt bind-oljuk!

        print(f"  EXCEL SUB-AGENT: Dedicated LLM for Excel branch (model: {MODEL_FOR_EXCEL_BRANCH}) "
              f"initialized and bound with PythonInterpreterTool.")
    except Exception as e_init:
        print(f"  EXCEL SUB-AGENT: ERROR initializing dedicated LLM for Excel branch: {e_init}")
        return {
            **state,
            "excel_final_result_from_sub_agent": f"EXCEL_PROCESSING_FAILED: LLM init error for Excel - {str(e_init)[:100]}",
            "excel_processing_active": False,
            "excel_processing_status": "critical_error_excel_llm_init"
        }

    excel_system_prompt = """You are a specialized AI assistant for processing Excel files using Python's pandas library.
Your goal is to answer the user's question based on the content of an Excel file.
The user's original question regarding this Excel file is: "{excel_question}"
The Excel file is located at: "{excel_file_path}"
You have access to a `python_interpreter` tool. You MUST generate Python code snippets for this tool.

**Current State of Excel Processing:**
- Iteration within Excel sub-task: {excel_iter_count} out of {max_excel_iter}
- Previous code execution history (last 3 attempts): {excel_history_summary}

**Strategy for Excel Processing (Iterative Approach):**
1.  **Initial Inspection (if `excel_code_execution_history` is empty or no useful info yet):**
    * Your FIRST STEP should be to generate Python code to inspect the data. This code MUST:
        * `import pandas as pd`
        * Read the Excel file: `df = pd.read_excel(r'{excel_file_path}')` (The path is ready for raw string)
        * Print column names: `print("COLUMNS:", df.columns.tolist())`
        * Print data types: `print("DATA_TYPES:\\n", df.dtypes)`
        * Print the first 5 rows: `print("HEAD:\\n", df.head())`
    * The `python_interpreter` will execute this. Do not attempt calculations yet.

2.  **Targeted Code Generation (after inspection):**
    * Examine the `excel_code_execution_history` for outputs from inspection code.
    * Based on this, and the original question, generate Python code to:
        * Select relevant columns.
        * Clean data IF NECESSARY (e.g., remove '$', ',', convert to numeric using `df['Col'] = df['Col'].astype(str).str.replace(r'[\$,]', '', regex=True).astype(float)`).
        * Perform calculations (sum, filter, etc.).
        * Print ONLY the final numerical or string answer to the original question, formatted as requested by the question.

3.  **Error Handling & Iteration:**
    * If a previous code execution resulted in a `Standard Error` (visible in history), analyze the error and your previous code. Generate a REVISED Python script to fix it.
    * If you are stuck after {max_excel_iter} attempts, your generated code should simply be: `print("EXCEL_PROCESSING_FAILED: Max iterations reached.")`

Your output MUST be a call to the `python_interpreter` tool with the `code_string` argument containing the Python code you generated.
If, based on the `excel_code_execution_history`, you are CERTAIN that a previous `print()` statement in a successful code execution ALREADY contains the final answer to the original question, then instead of calling a tool, output the special string "FINAL_EXCEL_ANSWER_IS_IN_HISTORY:" followed by a clear reference to which part of the history contains the answer (e.g., "Output of step X: 123.45").
"""
    current_excel_question = state['excel_original_question']
    current_excel_file_path = state['excel_file_to_process']
    excel_iter_count = state.get('excel_iteration_count',0)
    max_excel_iter = state.get('excel_max_sub_iterations', 5)

    # Összefoglaló az előzményekről, hogy ne legyen túl hosszú a prompt
    history_summary = []
    current_excel_history = state.get('excel_code_execution_history',
                                      [])  # Ezt használjuk az összefoglalóhoz és az üzenetekhez

    if current_excel_history:
        for i, (code, out) in enumerate(reversed(current_excel_history[-3:])):  # Utolsó 3
            # Helyes iterációs szám az összefoglalóban:
            actual_attempt_number_in_summary = (excel_iter_count - len(current_excel_history)) + (
                        len(current_excel_history[-3:]) - 1 - i) + 1
            # Vagy egyszerűbben, ha az excel_iter_count a *következő* iterációt jelöli:
            # actual_attempt_number_in_summary = excel_iter_count - (len(current_excel_history[-3:]) -1 - i)
            # De mivel az excel_iter_count a már lefutottakat számolja az execute_excel_python_code-ban,
            # az első (len(current_excel_history) - 1 - i) helyesebb lehet.
            # A lényeg, hogy az összefoglalóban az "Attempt X" konzisztens legyen.
            # Egyelőre maradjunk a korábbi logikádnál, vagy egyszerűsítsük:
            display_attempt_num = excel_iter_count - (
                        len(current_excel_history[-3:]) - 1 - i) if excel_iter_count > 0 else 1

            history_summary.append(
                f"Attempt {display_attempt_num}:\nCode: {code[:150]}...\nOutput: {str(out)[:200]}...")
    history_summary_str = "\n---\n".join(
        history_summary) if history_summary else "No previous attempts in this sub-task."

    prompt_formatted = excel_system_prompt.format(
        excel_question=current_excel_question,
        excel_file_path=current_excel_file_path,
        excel_iter_count=excel_iter_count,
        max_excel_iter=max_excel_iter,
        excel_history_summary=history_summary_str
    )

    # === ÜZENETEK ÖSSZEÁLLÍTÁSA A HELYES SORRENDDEL (ÚJ LOGIKA KEZDETE) ===
    # Ezt a részt cseréld le/illeszd be a meglévő messages összeállítás helyére.
    messages = [SystemMessage(content=prompt_formatted)]

    if current_excel_history:
        # Ha van előzmény, akkor kell egy "kezdeti" HumanMessage, ami ezt elindította.
        initial_user_prompt_for_history = "Please perform the initial inspection of the Excel file by generating the appropriate Python code to understand its structure (columns, data types, first few rows)."
        messages.append(HumanMessage(content=initial_user_prompt_for_history))

        for code_attempt, output_attempt in current_excel_history:
            tool_call_id_for_history = f"excel_hist_agent_tc_{uuid.uuid4().hex[:6]}"
            messages.append(AIMessage(
                content="",  # Az AIMessage contentje üres, ha tool_call van
                tool_calls=[{
                    "name": "python_interpreter",  # Ez az eszköz neve, amit az LLM "hívott"
                    "args": {"code_string": code_attempt},  # A kód, amit futtatni kért
                    "id": tool_call_id_for_history
                }]
            ))
            messages.append(ToolMessage(content=str(output_attempt), tool_call_id=tool_call_id_for_history))

    # Az aktuális HumanMessage, ami a következő kódot kéri
    excel_user_request_prompt_current = (
        "A rendszerutasítások és a rendelkezésre álló előzmények alapján (különösen az előző kód futásának kimenete alapján), "
        "kérlek, add meg a következő Python kódot a `python_interpreter` eszköz számára a feladat megoldásához. "
        "Vagy jelezd, ha a válasz már megtalálható az előzményekben."
    )
    messages.append(HumanMessage(content=excel_user_request_prompt_current))
    # === ÜZENETEK ÖSSZEÁLLÍTÁSA VÉGE ===

    print(
        f"  EXCEL SUB-AGENT: Invoking DEDICATED LLM for Excel (model: {MODEL_FOR_EXCEL_BRANCH}). Number of messages: {len(messages)}.")
    if messages:
        print(
            f"  EXCEL SUB-AGENT: Last message type: {type(messages[-1])}, content preview: {messages[-1].content[:200]}...")
        # Részletesebb logolás a messages listáról, ha szükséges:
        # for i, msg_item in enumerate(messages):
        #     print(f"    DEBUG MSG {i}: Type={type(msg_item)}, Content='{str(msg_item.content)[:100]}...', ToolCalls={hasattr(msg_item, 'tool_calls') and msg_item.tool_calls}")

    # Itt már nem a globális 'llm'-et hívjuk, hanem az 'excel_llm_instance_for_this_node'-ot!
    try:
        ai_msg_excel: AIMessage = excel_llm_instance_for_this_node.invoke(messages)
    except Exception as e:
        print(f"  EXCEL SUB-AGENT: ERROR during DEDICATED LLM invoke: {e}")
        # Részletes traceback a hibakereséshez
        import traceback
        print(traceback.format_exc())
        return {
            **state,
            "excel_final_result_from_sub_agent": f"EXCEL_PROCESSING_FAILED: Dedicated LLM invocation error in sub-agent - {str(e)[:200]}",
            "excel_processing_active": False,
            "excel_processing_status": "critical_error_llm_invoke"
        }

    # Az LLM válaszának feldolgozása (tool_calls vagy FINAL_ANSWER_IN_HISTORY) innen változatlanul folytatódik,
    # ahogy a te meglévő kódodban van:
    if ai_msg_excel.tool_calls and len(ai_msg_excel.tool_calls) > 0:
        tool_call = ai_msg_excel.tool_calls[0]
        if tool_call['name'] == "python_interpreter" and 'code_string' in tool_call['args']:
            generated_code = tool_call['args']['code_string']
            print(f"  EXCEL SUB-AGENT: LLM generated code: {generated_code[:300]}...")
            return {
                **state,
                "excel_current_pandas_code": generated_code,
                "excel_processing_status": "awaiting_excel_code_execution"
            }
        else:
            print(f"  EXCEL SUB-AGENT: ERROR - LLM proposed unexpected tool: {tool_call['name']}")
            return {**state,
                    "excel_final_result_from_sub_agent": f"EXCEL_PROCESSING_FAILED: LLM proposed unexpected tool '{tool_call['name']}'.",
                    "excel_processing_active": False, "excel_processing_status": "critical_error_llm_tool"}


    elif ai_msg_excel.content and "FINAL_EXCEL_ANSWER_IS_IN_HISTORY:" in ai_msg_excel.content:
        answer_info_from_llm = ai_msg_excel.content.split("FINAL_EXCEL_ANSWER_IS_IN_HISTORY:", 1)[1].strip()
        # Javaslat: A system promptot úgy módosítsd, hogy az LLM magát a TÉNYLEGES VÁLASZT adja vissza
        # a "FINAL_ANSWER_IN_HISTORY:" után, ne csak egy utalást.
        # Például: "FINAL_ANSWER_IN_HISTORY: 89706.00"
        # Ha ez így van, akkor:
        determined_answer = answer_info_from_llm  # Feltéve, hogy az LLM már a konkrét értéket adja itt

        # Ha továbbra is az előzményekből kell "kibányászni" a logod alapján:
        if "Output of step" in answer_info_from_llm or determined_answer == "Could not determine from history. LLM should print final answer.":  # Vagy ha az LLM nem adta meg a konkrét értéket
            determined_answer = "Could not extract specific answer from history based on LLM's reference."  # Alapértelmezett, ha a referencia nem egyértelmű
            current_excel_history = state.get('excel_code_execution_history', [])
            if current_excel_history:
                for _, output_hist_str in reversed(current_excel_history):
                    # Csak akkor vesszük figyelembe, ha sikeres volt a futás
                    if "Standard Error" not in str(output_hist_str) and \
                        ("Script exited with return code:" not in str(output_hist_str) or "return code: 0" in str(output_hist_str)):
                        match_output = re.search(r"Standard Output:\n(.*)", str(output_hist_str), re.DOTALL)
                        if match_output:
                            clean_output = match_output.group(1).strip()
                            lines = clean_output.splitlines()
                            if lines:
                                last_line = lines[-1].strip()
                                # Próbáljuk meg a numerikus értéket megtalálni, ha az az utolsó sor
                                try:
                                    # Egyszerűsített kinyerés: ha az utolsó sor csak egy szám (és esetleg $ jel)
                                    potential_answer = re.sub(r'[^\d\.]', '', last_line)  # Csak számokat és pontot hagy
                                    if potential_answer:
                                        float(potential_answer)  # Ellenőrizzük, hogy valid szám-e
                                        determined_answer = potential_answer
                                        print(f"  EXCEL SUB-AGENT: Extracted numerical answer from last history output: {determined_answer}")
                                        break
                                except ValueError:
                                    # Ha az utolsó sor nem csak egy szám, használjuk az LLM által adott infót, ha az jobb
                                    if answer_info_from_llm and not "Output of step" in answer_info_from_llm:
                                        determined_answer = answer_info_from_llm  # Ha az LLM konkrét értéket adott
                                    else:
                                        determined_answer = f"Last clean output (may not be final format): {last_line}"
                            break
                if determined_answer == "Could not extract specific answer from history based on LLM's reference." and not current_excel_history:
                    determined_answer = "No history to extract answer from, despite LLM claim."

        print(f"  EXCEL SUB-AGENT: LLM indicates answer is in history. LLM raw info: '{answer_info_from_llm}'. Determined/Extracted answer: '{determined_answer}'")
        return {
            **state,
            "excel_final_result_from_sub_agent": determined_answer,
            # Itt a (remélhetőleg) kinyert/LLM által adott válasz van
            # "excel_processing_active": False, # <<< EZT A SORT VEDD KI! A finalize_excel_processing végzi a lezárást.
            "excel_processing_status": "final_answer_from_history"  # Ez a státusz jelzi a routernek, mit tegyen
        }
    else:
        print(
            f"  EXCEL SUB-AGENT: ERROR - LLM response unclear or did not request a tool. Content: '{ai_msg_excel.content}'")
        return {**state,
                "excel_final_result_from_sub_agent": f"EXCEL_PROCESSING_FAILED: LLM unclear response in sub-agent. Content: {ai_msg_excel.content}",
                "excel_processing_active": False, "excel_processing_status": "critical_error_llm_unclear"}

def execute_excel_python_code(state: AgentState) -> AgentState:
    print(f"--- EXCEL SUB-AGENT (Iter: {state.get('excel_iteration_count', 0)}): EXECUTING GENERATED PYTHON CODE ---")
    code_to_run = state.get('excel_current_pandas_code')

    if not code_to_run:
        print("  EXCEL SUB-AGENT: ERROR - No pandas code found in state to execute.")
        return {**state, "excel_final_result_from_sub_agent": "EXCEL_PROCESSING_FAILED: No code to execute.", "excel_processing_active": False, "excel_processing_status": "critical_error_no_code"}

    # A globális `tools` listából keressük a python_interpreter-t
    python_tool = next((t for t in tools if t.name == "python_interpreter"), None)
    if not python_tool:
        print("  EXCEL SUB-AGENT: ERROR - PythonInterpreterTool not found.")
        return {**state, "excel_final_result_from_sub_agent": "EXCEL_PROCESSING_FAILED: PythonInterpreterTool not found.", "excel_processing_active": False, "excel_processing_status": "critical_error_no_tool_instance"}

    print(f"  EXCEL SUB-AGENT: Attempting to execute code: {code_to_run[:400]}...")
    # A PythonInterpreterTool timeout_seconds argumentumát a tool maga kezeli (alapértelmezett 30s)
    tool_output_str = python_tool.invoke({"code_string": code_to_run})
    print(f"  EXCEL SUB-AGENT: Code execution raw output: {tool_output_str[:400]}...")


    new_history_entry = (code_to_run, tool_output_str)
    updated_history = state.get('excel_code_execution_history', []) + [new_history_entry]

    return {
        **state,
        "excel_code_execution_history": updated_history,
        "excel_iteration_count": state.get('excel_iteration_count', 0) + 1,
        "excel_current_pandas_code": None, # Fontos törölni, hogy a következő tervezésnél újra generálódjon
        "excel_processing_status": "excel_code_executed"
    }

def decide_next_excel_step(state: AgentState) -> str:
    print(f"--- EXCEL SUB-AGENT (Iter: {state.get('excel_iteration_count',-1)}): DECIDING NEXT STEP ---") # -1 hogy lássuk ha nincs még iteráció
    current_excel_iter = state.get('excel_iteration_count', 0)
    max_excel_iter_for_sub = state.get('excel_max_sub_iterations', 5)
    current_excel_status = state.get("excel_processing_status")

    if current_excel_status == "final_answer_from_history" or \
       current_excel_status == "critical_error_no_file" or \
       current_excel_status == "critical_error_llm_tool" or \
       current_excel_status == "critical_error_llm_unclear" or \
       current_excel_status == "critical_error_no_code" or \
       current_excel_status == "critical_error_no_tool_instance":
        print(f"  EXCEL SUB-AGENT: Critical error or final answer identified by LLM ({current_excel_status}). Finalizing.")
        return "route_to_finalize_excel" # Kilépés az Excel al-ágból

    last_code_output = ""
    if state.get('excel_code_execution_history'):
        _, last_code_output = state['excel_code_execution_history'][-1]

    # Ha az LLM által generált kód expliciten jelzi a hibát a max iteráció miatt
    if "EXCEL_PROCESSING_FAILED: Max iterations reached." in str(last_code_output):
        print("  EXCEL SUB-AGENT: LLM-generated code indicated max iterations. Finalizing.")
        # Frissítsük az állapotot, hogy a finalize tudja ezt
        # Ezt a finalize node-nak kellene kezelnie, de itt is jelezhetjük
        # state["excel_final_result_from_sub_agent"] = "EXCEL_PROCESSING_FAILED: Max iterations reached by LLM decision."
        # state["excel_processing_status"] = "max_iterations_by_llm"
        return "route_to_finalize_excel"

    if current_excel_iter >= max_excel_iter_for_sub:
        print(f"  EXCEL SUB-AGENT: Max iterations ({max_excel_iter_for_sub}) for sub-agent reached.")
        # state["excel_final_result_from_sub_agent"] = f"EXCEL_PROCESSING_FAILED: Max {max_excel_iter_for_sub} iterations in sub-agent."
        # state["excel_processing_status"] = "max_iterations_hard_limit"
        return "route_to_finalize_excel"

    # Ha a kód lefutott, és még nem értük el a max iterációt, és az LLM nem adott végső választ,
    # akkor újra tervezünk az LLM-mel.
    if current_excel_status == "excel_code_executed":
        # Itt megvizsgálhatnánk a last_code_output-ot, hogy az LLM szerint ez már a válasz-e
        # De ezt az `excel_llm_planner_coder`-re bízzuk.
        print("  EXCEL SUB-AGENT: Code executed, returning to LLM for analysis/next plan.")
        return "route_to_excel_llm_planner" # Vissza az Excel LLM-hez

    # Alapértelmezett eset (pl. az `prepare_excel_processing_state` után, amikor `status` = "starting_excel_processing")
    print(f"  EXCEL SUB-AGENT: Defaulting to LLM planning. Status: {current_excel_status}")
    return "route_to_excel_llm_planner"

def finalize_excel_processing(state: AgentState) -> AgentState:
    print(f"--- EXCEL SUB-AGENT (Iter: {state.get('excel_iteration_count',0)}): FINALIZING ---")

    # Meghatározzuk a végeredményt az Excel al-ágból
    excel_result = state.get('excel_final_result_from_sub_agent')
    current_status = state.get('excel_processing_status')
    current_iter = state.get('excel_iteration_count', 0)
    max_iter = state.get('excel_max_sub_iterations', 5)

    if not excel_result: # Ha még nincs explicit eredmény
        if current_status == "max_iterations_hard_limit" or current_iter >= max_iter :
            excel_result = f"EXCEL_PROCESSING_FAILED: Max iterations ({current_iter}/{max_iter}) reached in sub-agent."
        elif current_status == "max_iterations_by_llm":
             excel_result = "EXCEL_PROCESSING_FAILED: Max iterations reached by LLM decision in sub-agent."
        elif state.get('excel_code_execution_history'):
            # Ha nincs explicit eredmény, de volt kód futtatás, próbáljuk meg az utolsó outputot
            # (de ez lehet hibás is)
            _, last_output_str = state['excel_code_execution_history'][-1]
            if "EXCEL_PROCESSING_FAILED: Max iterations reached." in last_output_str:
                excel_result = last_output_str
            elif "Standard Error" in last_output_str or "Script exited with return code:" in last_output_str : #hiba az utolsó futásnál
                excel_result = f"EXCEL_PROCESSING_FAILED: Last attempt had error. Output: {last_output_str[:200]}"
            else: # Utolsó esély: az utolsó stdout
                match = re.search(r"Standard Output:\n(.*)", last_output_str, re.DOTALL)
                if match:
                    excel_result = match.group(1).strip()
                    if not excel_result: # Ha üres volt a stdout
                         excel_result = "EXCEL_PROCESSING_ENDED: Last output was empty."
                else: # Ha nem volt Standard Output rész
                    excel_result = f"EXCEL_PROCESSING_ENDED: Could not parse final output. Last raw: {last_output_str[:200]}"
        else:
            excel_result = "EXCEL_PROCESSING_FAILED: Unknown state or no output produced."

    print(f"  EXCEL SUB-AGENT: Final result determined as: {excel_result}")

    main_intermediate_steps = state.get('intermediate_steps', [])
    # Egyértelműen jelezzük, hogy ez az Excel al-ág kimenete
    excel_sub_agent_output_entry = ("excel_sub_agent_result", excel_result)
    updated_main_intermediate_steps = main_intermediate_steps + [excel_sub_agent_output_entry]

    # Döntés arról, hogy a `final_answer` beállításra kerüljön-e
    is_successful_excel_result = "FAILED" not in excel_result.upper() and \
                                 "ERROR" not in excel_result.upper() and \
                                 len(excel_result.strip()) > 0 and \
                                 "COULD NOT PARSE" not in excel_result.upper() and \
                                 "UNKNOWN STATE" not in excel_result.upper() and \
                                 "EMPTY" not in excel_result.upper()


    return {
        **state,
        "intermediate_steps": updated_main_intermediate_steps,
        "final_answer": excel_result if is_successful_excel_result else None,
        "error_message": None if is_successful_excel_result else (state.get("error_message") or excel_result), # Csak akkor írjuk felül, ha hiba van
        "excel_processing_active": False, # Nagyon fontos, hogy itt False legyen!
        # Az Excel-specifikus mezőket itt nullázhatjuk, ha akarjuk, de nem kötelező
        # "excel_original_question": None,
        # "excel_file_to_process": None,
        # "excel_current_pandas_code": None,
        # "excel_code_execution_history": [], # Ezt meghagyhatjuk debuggoláshoz
        # "excel_processing_status": "completed",
        # "excel_iteration_count": 0,
        "excel_final_result_from_sub_agent": None # Ezt már feldolgoztuk
    }


def plan_next_step(state: AgentState) -> AgentState:
    """Az LLM-et hívja meg, hogy megtervezze a következő lépést vagy a választ."""
    print("--- AGENT PLANNING (LLM) ---")

    tool_descriptions_list = []
    for tool_instance in tools:  # Feltételezve, hogy 'tools' a globális eszközlista neve
        tool_descriptions_list.append(f"- **{tool_instance.name}**: {tool_instance.description}")
    tool_descriptions_str = "\n".join(tool_descriptions_list)

    question = state['original_question']
    task_id = state['task_id']
    intermediate_steps = state['intermediate_steps']
    file_path_from_state = state.get('file_path') # Erre a tesztblokkban nincs közvetlenül szükség,
    # de a normál logika később használhatja
    current_iterations = state.get('iterations', 0)
    MAX_ITERATIONS_LIMIT = 15  # Állíts be egy ésszerű limitet

    if current_iterations >= MAX_ITERATIONS_LIMIT:
        print(f"  Max iterations ({MAX_ITERATIONS_LIMIT}) reached. Providing error and stopping.")
        return {
            **state,
            'final_answer': "A folyamat leállt a maximális iterációszám elérése miatt.",
            'error_message': "MAX_ITERATIONS_REACHED",
            'iterations': current_iterations  # Vagy current_iterations + 1
        }

    print(f"  Iteration: {current_iterations + 1}")
    print(f"  Question: {question}")
    print(f"  Task ID: {task_id}")
    # print(f"  Current File path: {file_path}") # Ezt most kikommentezhetjük, mert a tesztben code_string-et használunk
    downloaded_file_info_str = ""
    if file_path_from_state:  # Ez az állapotból jön, amit az execute_tool frissít
        downloaded_file_info_str = f"CONTEXT_FROM_AGENT_STATE: A file for task_id '{task_id}' has already been successfully downloaded and is available at path: '{file_path_from_state}'. You should use this path directly with tools like 'python_interpreter' if needed. DO NOT use 'file_downloader' again for this task_id."
    else:
        # Ellenőrizzük az intermediate_steps-et is, hátha ott van már sikeres letöltés
        # (bár ideális esetben a file_path_from_state már tartalmazná)
        last_fd_success_path = None
        for step_tool, step_output in reversed(intermediate_steps):
            if step_tool == "file_downloader" and "File downloaded successfully. Available at path:" in str(
                    step_output):
                try:
                    last_fd_success_path = str(step_output).split("Available at path: ")[1]
                    break
                except IndexError:
                    pass  # Hiba a path parse-olásakor, hagyjuk
        if last_fd_success_path:
            downloaded_file_info_str = f"CONTEXT_FROM_INTERMEDIATE_STEPS: A file for task_id '{task_id}' was successfully downloaded previously and is available at path: '{last_fd_success_path}'. You should use this path. DO NOT use 'file_downloader' again for this task_id."
            # Frissíthetnénk itt az state['file_path']-t is, ha még None volt, de a promptba elég ez az infó.
        else:
            downloaded_file_info_str = "CONTEXT_FROM_AGENT_STATE: No file has been confirmed as downloaded yet for this task_id, or a previous download attempt failed with 404 (check intermediate_steps)."

    system_prompt_template = """You are a highly capable AI assistant for the GAIA Level 1 benchmark.
    Your goal is to accurately answer the user's question. Today's date is {current_date}.
    The user's question is for task_id: {task_id}.
    {downloaded_file_context_for_llm}

    You have access to the following tools:
    {tool_descriptions}
    
    **WEB SEARCHING (IMPORTANT!):**
    If you need to perform a web search to find current information, facts, or details not present in your knowledge base,
    you MUST indicate this by outputting a JSON object string with the following exact structure, and nothing else:
    ```json
    {{
      "action": "INITIATE_SEARCH_SUB_GRAPH",
      "initial_query": "your initial concise search query here"
    }}
    ```
    This is the **only** method you should use for general web searching.
    The search sub-graph will handle the search execution and result analysis.
    Do NOT attempt to use any of the tools from the list above for general web searching.
    The search sub-graph will handle the search execution and result analysis.
    
    **SPECIAL INSTRUCTION FOR EXCEL/CSV FILES:**
    If the user's question requires processing an Excel (.xlsx) or CSV (.csv) file, AND a file has been downloaded (its `file_path` is available in the context),
    your primary action should be to request specialized Excel processing.
    To do this, you MUST call a "virtual" tool named `REQUEST_EXCEL_PROCESSING`.
    Provide a brief `reason` in the arguments for this tool call.
    Example tool call:
    {{
      "name": "REQUEST_EXCEL_PROCESSING",
      "args": {{
        "reason": "The question asks to sum sales data from the provided Excel sheet."
      }}
    }}
    DO NOT attempt to generate Python code directly for Excel/CSV files in this main planning step.
    Let the specialized Excel processing branch handle it.
    
    Based on the user's question and the history of tool calls and their outputs (intermediate steps), decide the next action.
    You can either:
    1. Call one of the available tools from the list above (if not for web search or Excel).
    2. Output the JSON action string to INITIATE_SEARCH_SUB_GRAPH if web search is needed.
    3. Output the JSON action string to REQUEST_EXCEL_PROCESSING if Excel processing is needed.
    4. Directly answer the question if you have sufficient information. Provide only the answer itself.
    
    CRITICAL INSTRUCTIONS FOR FILE HANDLING AND TOOL USAGE:
    1.  **FILE DOWNLOADER (`file_downloader`):**
        * If the question implies a file is needed AND the context above indicates NO file has been successfully downloaded yet for this task_id (and no previous 404 error for it), THEN call `file_downloader` with the `task_id`.
        * If the context (either `CONTEXT_FROM_AGENT_STATE` or `CONTEXT_FROM_INTERMEDIATE_STEPS`) already shows a successfully downloaded `file_path` for this `task_id`, **DO NOT USE `file_downloader` AGAIN.** Use the existing path.
        * If `file_downloader` previously returned a '404 File Not Found' error for this `task_id`, **DO NOT USE `file_downloader` AGAIN.** Assume the file is unavailable.

    2.  **PYTHON INTERPRETER (`python_interpreter`):**
        * Use this tool for the following purposes:
            * To execute Python script files (.py) if one is provided or directly relevant (e.g., after being downloaded).
            * To execute SHORT, self-contained Python code snippets for general calculations or text manipulations, IF AND ONLY IF these do NOT involve processing complex file structures like Excel/CSV directly.
        * **DO NOT use this `python_interpreter` tool directly in THIS planning step to write or execute pandas code for Excel/CSV files.** If Excel/CSV processing is needed, use the `REQUEST_EXCEL_PROCESSING` signal.
        * **Input for executing .py scripts:**
            * If a .py script was downloaded, provide its `file_path` to this tool using the 'file_path' argument.
        * **General:**
            * This tool has a `timeout_seconds` argument (default 30s, can be increased to 90s once if timeout occurs).
            * The tool returns standard output and standard error. Use any printed output to answer the question. If it reports `ModuleNotFoundError` for `pandas`, it means the library is not available, and you should state that processing the file is not possible.

    3.  **WEB SEARCHING STRATEGY (IMPORTANT - Follow Carefully!):**

        To perform a general web search for current information, facts, or details not present in your knowledge base, your **PRIMARY and PREFERRED method** is to use a dedicated external Google Search script.
        To trigger this, your response **MUST BE A JSON OBJECT STRING, AND NOTHING ELSE,** with the following exact structure:
        ```json
        {{
          "action": "INITIATE_SEARCH_SUB_GRAPH",
          "initial_query": "your initial concise search query here"
        }}
        ```
        **Example of when to use this:** If the user asks "What is the current weather in London?" or "Latest news about a specific topic", you should output the JSON above with the appropriate query.
    
        **CRITICAL:**
        *   When using the `INITIATE_SEARCH_SUB_GRAPH` action, your entire output must be ONLY the JSON string. No other text.
        *   Prioritize `INITIATE_SEARCH_SUB_GRAPH` for general, up-to-date information retrieval.
        
    4.  **MULTIMODAL FILE PROCESSOR (`multimodal_file_processor`):**
        * If the question requires understanding the content of an audio, image, or video file, use this tool.
        * **Step 1:** Ensure the file is downloaded using `file_downloader`. Get the `file_path`.
        * **Step 2:** Call `multimodal_file_processor` with the following arguments:
            * `file_path`: The path of the downloaded file (e.g., 'temp_gaia_files/your_file.mp3').
            * `user_prompt`: A clear instruction on what to do with the file (e.g., "Transcribe this audio.", "Describe the objects in this image and any text visible.", "What is the FEN notation for this chess position? It is black's turn.").
            * `mime_type` (optional but helpful): The MIME type of the file, such as 'audio/mpeg' for .mp3, 'image/png' for .png, 'image/jpeg' for .jpg, 'video/mp4' for .mp4. If you omit this, the tool will try to guess from the file extension, but providing it is safer.
        * This tool will return the LLM's textual analysis of the file content (e.g., a transcript, a description). Use this output for subsequent reasoning or to form the final answer.
        
    5.  **CHESS POSITION ANALYZER (`chess_position_analyzer`):**
        * Use this tool ONLY AFTER you have obtained a FEN string representing the chess position.
        * **Input:** Requires a `fen` argument (the FEN string).
        * **Action:** This tool uses a chess engine (Stockfish) to analyze the position from the FEN string and find the best or winning move for the current player indicated in the FEN.
        * **Output:** The suggested move in algebraic notation (e.g., "Qh8#", "Nf3").
        * **Workflow for chess image questions:**
            1. Use `file_downloader` to download the image.
#           2. Use `external_image_to_fen_generator` with the image `image_file_path` to get the piece placement FEN.
#           3. **Crucial Step:** Examine the question to determine whose turn it is (e.g., "black's turn"). Modify the FEN string received from `external_image_to_fen_generator` to reflect the correct player to move (e.g., change 'w' to 'b'). The other parts of the FEN (castling, en passant, move counters) from the generator can usually be kept as is or set to defaults like '- - 0 1' if unsure and the generator does not provide them.
#           4. Take the corrected, full FEN string and pass it to `chess_position_analyzer` using the `fen` argument.
#           5. The output of `chess_position_analyzer` should be the final answer if it's a valid move.

    6.  **EXTERNAL IMAGE TO FEN GENERATOR (`external_image_to_fen_generator`):**
        * If the question involves a chess diagram image and you need the FEN string, use this specialized tool.
        * **Step 1:** Ensure the image file is downloaded using `file_downloader`. Get the `file_path`.
        * **Step 2:** Call `external_image_to_fen_generator` with:
           * `image_file_path`: The absolute path of the downloaded image file.
        * This tool will return the FEN string representing the piece placement. This FEN might need adjustment for whose turn it is before passing to `chess_position_analyzer`.
        
    7.  **YOUTUBE VIDEO ANALYZER (`youtube_video_analyzer`):**
        *   If the user's question explicitly provides a YouTube video URL (e.g., `https://www.youtube.com/watch?v=...` or `https://youtu.be/...`) AND the question requires understanding the *content* of that video (e.g., summarizing it, finding specific information said or shown in it, answering a question based on the video's dialogue or visuals), then you SHOULD use this `youtube_video_analyzer` tool.
        *   **Input for this tool:**
            *   `video_url`: The full YouTube video URL from the question.
            *   `user_prompt`: A clear and specific instruction or question for the AI about what to extract or understand from the video. For example: "Transcribe the exact dialogue where the question 'Isn't that hot?' is asked and Teal'c's immediate response.", or "Find the most relevant comment or part of the video description that answers what Teal'c says in response to 'Isn't that hot?'".
        *   **Output:** This tool will return a textual response from an AI model that has attempted to analyze the video's content based on your prompt.
        *   **IMPORTANT Considerations:**
            *   This tool directly processes the video via its URL. Do NOT use `file_downloader` for YouTube URLs if you intend to analyze the video's content with this tool.
            *   If the question asks to download a *separate file* that is merely *linked* in a YouTube video's description (e.g., a PDF or a dataset), then you should first use `web_search` or `youtube_video_analyzer` (with a prompt asking for the description text) to find that specific file's download URL, and then potentially use `file_downloader` if it's a direct file link (not another webpage).
            *   If this `youtube_video_analyzer` tool fails to extract the needed information or returns an error (e.g., video inaccessible, content too complex), you might then consider using `web_search` to find an existing transcript or summary of the video online, which you could then analyze.
            *   For analyzing downloaded video *files* (e.g., .mp4 files obtained via `file_downloader` from the GAIA API, if such tasks exist), you would use `multimodal_file_processor` (for audio transcription from the video file) or `python_interpreter` (with `ffmpeg` if you need to extract frames/audio first, though this is more complex). Prioritize `youtube_video_analyzer` for direct YouTube URL analysis.

    GENERAL STRATEGY:
    - Prioritize using an existing downloaded file path.
    - If a tool fails (e.g., timeout or `ModuleNotFoundError` from `python_interpreter`), note the error. If it's a missing library for Python code, you cannot fix it, so state that the code couldn't be run. For other tool failures, consider if an alternative tool or approach is viable.
    - After a maximum of 2-3 attempts with `python_interpreter` (including timeout adjustments), if no result is obtained, provide a final answer stating what was attempted and why the result could not be obtained.
    - Your goal is to answer the question. If you have the answer, provide it directly.
    - If `youtube_video_analyzer` was used for a YouTube video but the returned information seems insufficient or doesn't directly answer the question, consider using `web_search` with a query like "What did Teal'c say to 'Isn't that hot?' in Stargate SG-1 YouTube video [video_title_or_id]" to find transcripts, summaries, or discussions about that specific scene.
    
    FINAL ANSWERING PROTOCOL:
    - Answer as briefly as possible unless a characteristic is part of the answer. e.g. properties of ingredients.
    - If a tool execution (especially `python_interpreter`, `multimodal_file_processor`, or `chess_position_analyzer`) provides a direct and complete answer to the original question, your next action MUST be to provide this answer as the final response.
    - If `chess_position_analyzer` returns a specific chess move (e.g., "Rd5", "Qh8#"), THIS IS THE FINAL ANSWER. Provide this move directly. DO NOT attempt further analysis or tool calls.
    - If `chess_position_analyzer` returns a string starting with "FINAL_CHESS_MOVE: ", extract the move part and provide it as the final answer.
    - Do NOT call additional tools or try to re-evaluate if a previous step has already yielded a clear answer. For example, if `python_interpreter` outputs a number and the question asks for a numeric output, that number IS the final answer. Similarly, if `chess_position_analyzer` provides a move, that move IS the final answer.
    - Only call further tools if the output of a previous tool needs significant interpretation, combination with other information, or formatting that you cannot do directly.
    - Capitalize the first letter of the word right.
    - Do not use abbreviations; for example, write 'Saint' instead of 'St.'.
    - Write the ingredients with lowercase letters.
    - In your answer, write the numbers numerically, not as strings, without units.
    - In your answer, omit the full stops and other punctuation marks.
    - If a tool execution OR an `excel_sub_agent_result` from `intermediate_steps` provides a direct and complete numerical answer to the original question, your next action MUST be to provide this answer as the final response. Format it as requested by the original question (e.g., USD with two decimal places).
    """


    # LLM-et "összekötjük" az eszközökkel
    # Fontos, hogy a `tools` lista itt azokat az eszközöket tartalmazza, amiket tényleg használni akarunk.
    llm_with_tools = llm.bind_tools(tools)  # Ezt a sort hagyd meg, ahol eredetileg volt, vagy tedd ide.

    # --- ÚJ: Intermediate Steps Összefoglalása és Üzenetek Összeállítása ---
    agent_scratchpad_messages = []
    if intermediate_steps:
        MAX_HISTORY_ITEMS_FOR_LLM = 15  # Csak az utolsó N elemet vesszük figyelembe
        recent_steps = intermediate_steps[-MAX_HISTORY_ITEMS_FOR_LLM:]  # Csak az utolsó N lépés

        for tool_call_idx, (tool_name, tool_output) in enumerate(recent_steps):
            dummy_tool_call_id = f"tool_call_intermediate_{tool_name}_{len(intermediate_steps) - len(recent_steps) + tool_call_idx}_{uuid.uuid4().hex[:6]}"

            agent_scratchpad_messages.append(AIMessage(
                content="",
                tool_calls=[{
                    "id": dummy_tool_call_id,
                    "name": tool_name,
                    "args": {}  # Egyszerűsítettük, az LLM a tool_output-ra támaszkodik
                }]
            ))

            MAX_TOOL_OUTPUT_LENGTH = 1500  # Karakterlimit az eszköz kimenetére
            tool_output_str = str(tool_output)
            if len(tool_output_str) > MAX_TOOL_OUTPUT_LENGTH:
                # Ha a kimenet hibát tartalmaz (pl. traceback), akkor azt ne vágjuk le nagyon,
                # mert fontos lehet az LLM-nek. Próbáljuk meg a hibát megtartani.
                # Ez egy egyszerűsített ellenőrzés, finomítható.
                error_keywords = ["Error", "Traceback", "Exception", "KeyError", "ValidationError"]
                contains_error = any(keyword in tool_output_str for keyword in error_keywords)

                if contains_error:
                    # Hagyjunk többet a hibából, pl. az első 500 és az utolsó 500 karaktert
                    tool_output_summary = tool_output_str[
                                          :500] + "\n... (error output truncated) ...\n" + tool_output_str[-500:]
                    if len(tool_output_summary) > MAX_TOOL_OUTPUT_LENGTH * 1.5:  # Ha még így is túl hosszú
                        tool_output_summary = tool_output_str[
                                              :MAX_TOOL_OUTPUT_LENGTH] + "... (long error output truncated)"
                else:
                    tool_output_summary = tool_output_str[:MAX_TOOL_OUTPUT_LENGTH] + "... (output truncated)"
            else:
                tool_output_summary = tool_output_str

            agent_scratchpad_messages.append(ToolMessage(content=tool_output_summary, tool_call_id=dummy_tool_call_id))

    prompt_messages = [
        SystemMessage(content=system_prompt_template.format(
            current_date=CURRENT_DATE_STR,
            task_id=state['task_id'],
            downloaded_file_context_for_llm=downloaded_file_info_str,
            tool_descriptions=tool_descriptions_str
        )),
        HumanMessage(content=question),
    ]
    prompt_messages.extend(agent_scratchpad_messages)  # Itt már az összefoglalt/rövidített előzmények kerülnek be

    print(f"  Invoking LLM for planning... Number of messages for LLM: {len(prompt_messages)}")
    # Becsüljük meg a karaktereket (NEM tokeneket, csak egy durva indikátor)
    approx_chars = 0
    for msg in prompt_messages:
        if hasattr(msg, 'content') and msg.content:
            approx_chars += len(str(msg.content))
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                approx_chars += len(str(tc))  # Hozzáadjuk a tool_call reprezentáció hosszát is
    print(f"  Approximate character count for LLM input (sum of content and tool_calls): {approx_chars}")
    # --- Intermediate Steps Összefoglalása és Üzenetek Összeállítása VÉGE ---

    # Az LLM hívása és a válasz feldolgozása innen ugyanúgy folytatódik, ahogy eddig:
    try:
        # Győződj meg róla, hogy az `llm` (a ChatGoogleGenerativeAI példányod)
        # és a `tools` (a Pydantic-alapú tool-ok listája) helyesen van definiálva
        # és elérhető ebben a scope-ban.
        # Ha a `tools` lista üres, mert minden speciális action-nel van megoldva,
        # akkor a `bind_tools` nem feltétlenül szükséges, vagy egy üres listát kap.
        # Most feltételezem, hogy a `tools` lista tartalmazza a Pydantic toolokat.

        llm_to_invoke = llm  # Alapértelmezetten a nyers LLM
        if 'tools' in globals() and isinstance(tools, list) and tools:  # Ha vannak Pydantic toolok
            llm_to_invoke = llm.bind_tools(tools)
            print("  LLM bound with Pydantic tools for potential tool_calls.")
        else:
            print(
                "  No Pydantic tools to bind. Using raw LLM for invocation (expecting JSON actions or direct answer).")

        ai_msg: AIMessage = llm_to_invoke.invoke(prompt_messages)

    except Exception as e:
        print(f"  Error invoking LLM: {e}")
        import traceback  # Biztosítjuk az importot itt is
        traceback.print_exc()  # Részletesebb hiba a konzolra
        return {
            **state,
            'error_message': f"LLM_INVOCATION_ERROR: {str(e)[:200]}",
            'iterations': current_iterations + 1,
            'final_answer': "LLM_INVOCATION_FAILED"
        }

        # === LLM VÁLASZÁNAK FELDOLGOZÁSA ===
    print(f"  LLM raw response object: {ai_msg}")
    llm_content_str = ""
    if isinstance(ai_msg.content, str):
        llm_content_str = ai_msg.content.strip()
    elif isinstance(ai_msg.content, list):
        for item_in_content in ai_msg.content:
            if isinstance(item_in_content, str):
                llm_content_str += item_in_content + " "
            elif isinstance(item_in_content, dict) and "text" in item_in_content:
                llm_content_str += item_in_content["text"] + " "
        llm_content_str = llm_content_str.strip()
    print(f"  LLM response content (processed as string): '{llm_content_str}'")

    # 1. SPECIÁLIS JSON ACTION-ÖK ELLENŐRZÉSE
    processed_llm_content_for_json = llm_content_str
    if llm_content_str.startswith("```json"):
        processed_llm_content_for_json = llm_content_str.removeprefix("```json").removesuffix("```").strip()
    elif llm_content_str.startswith("```") and llm_content_str.endswith("```"):
        processed_llm_content_for_json = llm_content_str[3:-3].strip()

    if processed_llm_content_for_json and \
            processed_llm_content_for_json.startswith("{") and \
            processed_llm_content_for_json.endswith("}"):
        try:
            potential_json = json.loads(processed_llm_content_for_json)
            if isinstance(potential_json, dict):
                action_key_value = potential_json.get("action")
                name_key_value = potential_json.get("name") # Excel kéréshez

                if action_key_value == "INITIATE_SEARCH_SUB_GRAPH" and "initial_query" in potential_json:
                    print(f"  LLM action: INITIATE_SEARCH_SUB_GRAPH with query: {potential_json['initial_query']}")
                    return {
                        **state,
                        'current_tool_name': "INITIATE_SEARCH_SUB_GRAPH",
                        'current_tool_input': {"initial_query": potential_json['initial_query']},
                        'error_message': None, 'iterations': current_iterations + 1, 'final_answer': None
                    }
                elif name_key_value == "REQUEST_EXCEL_PROCESSING" and "args" in potential_json and "reason" in potential_json["args"]: # JAVÍTVA ITT
                    print(f"  LLM action: REQUEST_EXCEL_PROCESSING with reason: {potential_json['args'].get('reason')}")
                    return {
                        **state,
                        'current_tool_name': "REQUEST_EXCEL_PROCESSING", # Fontos jelzés a routernek
                        'current_tool_input': potential_json, # A teljes JSON objektumot átadjuk
                        'error_message': None, 'iterations': current_iterations + 1, 'final_answer': None # Nincs még végső válasz!
                    }
                # ... (többi, esetleges "action" vagy "name" alapú speciális JSON kezelése) ...
                else:
                    # Ha volt "action" vagy "name" kulcs, de nem ismertük fel, logoljuk és hagyjuk továbbmenni
                    if action_key_value:
                        print(f"  LLM content was JSON with an unrecognized 'action': '{action_key_value}'. Proceeding.")
                    elif name_key_value:
                        print(f"  LLM content was JSON with an unrecognized 'name': '{name_key_value}'. Proceeding.")
        except (json.JSONDecodeError, TypeError) as e:
            print(f"  DEBUG: LLM content JSON parsing error for '{processed_llm_content_for_json}'. Error: {e}. Proceeding.")

    # 2. NORMÁL LANGCHAIN TOOL_CALLS ELLENŐRZÉSE
    if ai_msg.tool_calls and len(ai_msg.tool_calls) > 0:
        print(f"  LLM wants to call standard LangChain tools: {ai_msg.tool_calls}")

        # Az ai_msg.tool_calls egy lista, amiben tool hívás dictionary-k vannak.
        # Csak az első tool hívást vesszük figyelembe a jelenlegi logikában.
        tool_call_to_process = ai_msg.tool_calls[0]  # <<< JAVÍTÁS ITT: az első elemet vesszük

        tool_name_to_call = tool_call_to_process.get('name')
        tool_input_to_call = tool_call_to_process.get('args', {})

        if not tool_name_to_call:
            error_msg_for_tool = "Invalid tool_call structure from LLM (missing 'name')."
            print(f"  Error: {error_msg_for_tool}")
            # ... (hibakezelés, mint korábban)
            return {**state, 'error_message': error_msg_for_tool, 'iterations': current_iterations + 1,
                    'final_answer': None, 'current_tool_name': None, 'current_tool_input': None}

        # Speciális ellenőrzés a "search_sub_graph_result" "tool" hívásra, hogy ne akadjon be
        if tool_name_to_call == "search_sub_graph_result":
            error_msg_for_tool = f"LLM attempted to call '{tool_name_to_call}' which is a result, not a callable tool. LLM should process this result from intermediate_steps."
            print(f"  Warning: {error_msg_for_tool}")
            updated_intermediate_steps = state.get('intermediate_steps', []) + [
                (tool_name_to_call, "ERROR: This is a result, not a callable tool.")]
            return {
                **state,
                'intermediate_steps': updated_intermediate_steps,
                'error_message': error_msg_for_tool,  # Ezt az LLM látni fogja
                'current_tool_name': None,
                'current_tool_input': None,
                'iterations': current_iterations + 1,
                'final_answer': None
            }

        if not any(t.name == tool_name_to_call for t in tools):
            error_msg_for_tool = f"LLM tried to call tool '{tool_name_to_call}' which is not in the defined tools list."
            print(f"  Error: {error_msg_for_tool}")
            updated_intermediate_steps = state.get('intermediate_steps', []) + [
                (tool_name_to_call, f"ERROR: Tool '{tool_name_to_call}' not found.")]
            return {
                **state,
                'intermediate_steps': updated_intermediate_steps,
                'error_message': error_msg_for_tool,
                'current_tool_name': None,
                'current_tool_input': None,
                'iterations': current_iterations + 1,
                'final_answer': None
            }
        else:
            print(f"  LLM selected actual tool: '{tool_name_to_call}' with args: {tool_input_to_call}")
            return {
                **state,
                'current_tool_name': tool_name_to_call,
                'current_tool_input': tool_input_to_call,
                'error_message': None,
                'iterations': current_iterations + 1,
                'final_answer': None
            }

    # 3. DIREKT VÁLASZ ELLENŐRZÉSE
    if llm_content_str and llm_content_str.strip():
        print("  LLM provided a direct answer.")
        return {
            **state,
            'final_answer': llm_content_str.strip(),
            'current_tool_name': None, 'current_tool_input': None, 'error_message': None,
            'iterations': current_iterations + 1
        }

    # 4. HA SEMMI MÁS
    print("  LLM did not provide a usable action, tool call, or direct answer.")
    return {
        **state,
        'final_answer': "LLM_EMPTY_RESPONSE_FINAL_FALLBACK",
        'error_message': "LLM_EMPTY_OR_INVALID_RESPONSE",
        'current_tool_name': None, 'current_tool_input': None,
        'iterations': current_iterations + 1
    }


def execute_tool(state: AgentState) -> AgentState:
    """Végrehajtja a kiválasztott eszközt."""
    print("--- AGENT EXECUTING TOOL ---")
    tool_name = state.get('current_tool_name')
    tool_input = state.get('current_tool_input')

    if not tool_name:
        print("  No tool selected to execute in current state.")
        # Nem feltétlenül hiba, lehet, hogy az LLM választ adott. A should_continue kezeli.
        # De ha idejutunk és nincs tool_name, akkor a graph rosszul van összerakva.
        # A should_continue_or_finish-nek ezt el kellene kapnia.
        state['error_message'] = "INTERNAL_GRAPH_ERROR: execute_tool called without a tool selected."
        return {**state}
    print(f"  Executing tool: {tool_name} with input: {tool_input}")

    selected_tool = next((t for t in tools if t.name == tool_name), None)

    if not selected_tool:
        error_msg = f"Error: Tool '{tool_name}' was selected by LLM but not found in the available tools list."
        print(error_msg)
        # Ezt az `intermediate_steps`-be tesszük, hogy az LLM lássa a következő körben
        return {
            **state,
            'intermediate_steps': state['intermediate_steps'] + [(tool_name, error_msg)],
            'current_tool_name': None,
            'current_tool_input': None,
            'error_message': error_msg  # Ezt a mezőt is beállíthatjuk, de az intermediate_steps fontosabb az LLM-nek
        }

    tool_result_str = ""
    try:
        # Az eszközök `invoke` metódusát használjuk.
        # Az inputnak dictionary-nek kell lennie, ha az args_schema van definiálva.
        if not isinstance(tool_input, dict) and selected_tool.args_schema:
            # Ez egy hack, jobb, ha az LLM mindig helyes dict-et ad.
            # Feltételezi, hogy az input a séma ELSŐ mezőjének értéke.
            # Langchain tool hívásnál az LLM általában a teljes { "arg_name": value } dict-et adja.
            # Ha mégsem, ez megpróbálja kezelni.
            print(f"  Warning: Tool input for {tool_name} was not a dict, attempting to wrap.")
            if selected_tool.args_schema.__fields__:
                first_field_name = list(selected_tool.args_schema.__fields__.keys())[0]
                tool_result = selected_tool.invoke({first_field_name: tool_input})
            else:  # Nincs definiált mező az args_schema-ban, de args_schema létezik? Fura.
                tool_result = selected_tool.invoke(tool_input)
        else:  # Az input dict (ahogy várjuk) vagy nincs args_schema
            print(f"  LLM generated tool_input for {tool_name}: {tool_input}")
            tool_result = selected_tool.invoke(tool_input if isinstance(tool_input, dict) else str(tool_input))

        tool_result_str = str(tool_result)  # Itt a tool_result a PythonInterpreterTool teljes kimenete kell, hogy legyen
        print(f"  Tool {tool_name} executed. Raw Result (first 500 chars for log): {tool_result_str[:500]}...")

        # Speciális kezelés a FileDownloaderTool eredményére
        updated_file_path = state.get('file_path')  # Alapértelmezetten a régi érték

        if tool_name == "file_downloader":

            if "File downloaded successfully. Available at path: " in tool_result_str:
                try:
                    parsed_path = tool_result_str.split("Available at path: ")[1].strip()
                    updated_file_path = parsed_path
                    print(f"  FileDownloader successful. State 'file_path' was updated to: {updated_file_path}")
                except IndexError:
                    # Hiba a path parse-olásakor, a tool_result_str már tartalmazhatja a hibát
                    print(f"  Error parsing path from FileDownloader success message: {tool_result_str}")



        # state['intermediate_steps'].append((tool_name, tool_result_str)) # Ezt már a state update-ben kezeljük
        return {
            **state,
            'intermediate_steps': state['intermediate_steps'] + [(tool_name, tool_result_str)],
            'current_tool_name': None,
            'current_tool_input': None,
            'file_path': updated_file_path, # Itt használjuk a frissített értéket
            'error_message': None if not ("Error:" in tool_result_str or "Script exited with return code:" in tool_result_str and "return code: 0" not in tool_result_str) else tool_result_str
}

    except Exception as e:
        error_msg = f"Error during execution of tool {tool_name}: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()  # Részletes hiba a konzolra
        return {
            **state,
            'intermediate_steps': state['intermediate_steps'] + [(tool_name, error_msg)],
            'current_tool_name': None,
            'current_tool_input': None,
            'error_message': error_msg
        }


def should_continue_or_finish(state: AgentState) -> str:
    print("--- AGENT DECIDING FLOW (MAIN ROUTER) ---")

    # Állapotváltozók lekérdezése a state-ből a függvény elején
    final_answer = state.get('final_answer')
    current_tool_name = state.get('current_tool_name')  # Ezt a változót használjuk végig
    current_main_error = state.get('error_message')

    # 1. VÉGSŐ VÁLASZ ESETÉN BEFEJEZÉS
    if final_answer:
        print(f"  Decision: Final answer ('{str(final_answer)[:100]}...') is present. Finishing.")
        return END

    # 2. KRITIKUS HIBÁK MIATTI BEFEJEZÉS
    if current_main_error:
        if "MAX_ITERATIONS_REACHED" in current_main_error:
            print(f"  Decision: Critical error (MAX_ITERATIONS_REACHED). Finishing.")
            return END
        if "LLM_EMPTY_OR_INVALID_RESPONSE" in current_main_error or \
                "LLM_NOT_AVAILABLE_FOR_PLANNING" in current_main_error:  # Egyéb kritikus hibák
            print(f"  Decision: Critical error ('{current_main_error}'). Finishing.")
            return END
        # Itt lehetne még más, futást megszakító error_message-eket is ellenőrizni

    # 3. SPECIÁLIS ACTION-ÖK / AL-ÁGAK INDÍTÁSA (MÓDOSÍTOTT RÉSZ)
    if current_tool_name == "INITIATE_SEARCH_SUB_GRAPH":
        print("  Decision: Routing to prepare_search_sub_graph node.")
        return "prepare_search_sub_graph"  # Ennek a stringnek meg kell egyeznie a gráf él definíciójában használt kulccsal

    if current_tool_name == "REQUEST_EXCEL_PROCESSING":  # << HOZZÁADVA
        print("  Decision: Routing to prepare_excel_processing node.")
        return "prepare_excel_processing"  # Ennek a stringnek meg kell egyeznie a gráf él definíciójában használt kulccsal

    # Régebbi keresési metódushoz tartozó feltétel, ellenőrizd a relevanciáját
    if current_tool_name == "PERFORM_EXTERNAL_GOOGLE_SEARCH":
        print("  Decision: Routing to external Google search script execution node ('call_external_search_node').")
        return "call_external_search_node"

    # 4. NORMÁL (PYDANTIC-ALAPÚ) ESZKÖZFUTTATÁS
    # Ha a current_tool_name be van állítva, és nem a fenti speciális action nevek egyike.
    if current_tool_name:
        print(
            f"  Decision: Normal LangChain tool '{current_tool_name}' selected. Executing tool via 'execute_tool' node.")
        return "execute_tool"

        # 5. ÚJRATERVEZÉS (HA NINCS KONKRÉT ACTION/TOOL, VAGY NEM KRITIKUS HIBA UTÁN)
    # Ha idáig eljutottunk, az azt jelenti:
    # - Nincs `final_answer`.
    # - Nincs olyan kritikus `current_main_error`, ami miatt kiléptünk volna.
    # - Nincs beállítva `current_tool_name` (sem speciális action, sem normál tool).
    # VAGY van `current_main_error`, de az nem volt kritikus (pl. egy tool futási hiba, amit az LLM-nek látnia kell).

    if current_main_error:
        print(
            f"  Decision: Non-critical error present ('{str(current_main_error)[:100]}...'). Re-planning via 'plan_next_step'.")
    else:
        print(
            f"  Decision: No specific action/tool selected, no final answer, no critical error. Re-planning via 'plan_next_step'.")
    return "plan_next_step"


def call_external_search_node(state: AgentState) -> AgentState:
    print("--- AGENT CALLING EXTERNAL GOOGLE SEARCH SCRIPT ---")
    search_input_dict = state.get('current_tool_input')
    query = search_input_dict.get("query") if isinstance(search_input_dict, dict) else None

    search_result_str = "Error: No query provided for external search."
    if query:
        # Az agent fő modelljét vagy a szkript alapértelmezettjét használhatjuk
        # A `model_id_for_search` a `GoogleNativeSearchTool` osztályban volt,
        # de most a szkriptnek adjuk át. Használhatunk egy konstansot vagy
        # az agent state-ből vesszük, ha ott tárolnánk.
        script_model_id = "gemini-2.5-flash-preview-05-20"  # Ezt használja az external_google_search_EXACT.py alapból
        search_result_str = execute_external_google_search_script(query, model_id_for_script=script_model_id)

    print(f"  External Google Search script result (first 300 chars): {search_result_str[:300]}...")

    tool_name_for_history = "external_google_search"

    # Fontos, hogy az error_message frissüljön, ha a szkript hibát adott vissza
    is_error = search_result_str.upper().startswith("ERROR")

    return {
        **state,
        'intermediate_steps': state['intermediate_steps'] + [(tool_name_for_history, search_result_str)],
        'current_tool_name': None,
        'current_tool_input': None,
        'error_message': search_result_str if is_error else None,  # Csak akkor állítjuk be, ha tényleg hiba
        'final_answer': None  # A keresés után újra tervezünk
    }


# --- Grafikon összeállítása ---
workflow = StateGraph(AgentState)

workflow.add_node("initialize_agent", initialize_agent)
workflow.add_node("plan_next_step", plan_next_step)
workflow.add_node("execute_tool", execute_tool)

workflow.add_node("call_external_search_node", call_external_search_node)

workflow.add_node("prepare_excel_processing", prepare_excel_processing_state)
workflow.add_node("excel_llm_planner_coder", excel_llm_planner_coder)
workflow.add_node("execute_excel_python_code", execute_excel_python_code)
workflow.add_node("finalize_excel_processing", finalize_excel_processing)

workflow.add_node("prepare_search_sub_graph", prepare_search_sub_graph)
workflow.add_node("search_query_refiner_or_planner_llm", search_query_refiner_or_planner_llm)
workflow.add_node("execute_native_search_node", execute_native_search_node) # Ez hívja a perform_native_google_search-t
workflow.add_node("finalize_search_sub_graph", finalize_search_sub_graph)
# A decide_next_search_step egy router, nem külön node, hanem feltételes élekben használjuk.

workflow.set_entry_point("initialize_agent")

workflow.add_edge("initialize_agent", "plan_next_step")

# Élek az Excel al-ágon belül:
workflow.add_edge("prepare_excel_processing", "excel_llm_planner_coder") # Előkészítés után -> Excel LLM tervező
def route_after_excel_planner(state: AgentState) -> str:
    status = state.get("excel_processing_status")
    print(f"  EXCEL ROUTER (after planner): Current status is '{status}'") # Logolás
    if status == "awaiting_excel_code_execution": # Az LLM kódot generált
        print("  EXCEL ROUTER (after planner): Routing to execute_excel_python_code")
        return "execute_code"
    elif status == "final_answer_from_history" or \
         status == "critical_error_llm_tool" or \
         status == "critical_error_llm_unclear" or \
         status == "critical_error_llm_invoke" or \
         status == "critical_error_empty_code" or \
         status == "critical_error_code_extraction": # Ha az LLM tervező hibát észlelt vagy a válasz megvan
        print(f"  EXCEL ROUTER (after planner): Routing to finalize_excel_processing due to status: {status}")
        return "finalize"
    else:
        # Váratlan állapot, vagy ha az LLM nem állított be egyértelmű státuszt
        print(f"  EXCEL ROUTER (after planner): Unexpected or initial status ('{status}'), assuming finalization or error.")
        # Biztonsági okokból ilyenkor is a finalize felé megyünk, hogy ne akadjon be.
        # A finalize_excel_processing majd kezeli, ha nincs konkrét excel_final_result_from_sub_agent.
        if state.get("excel_final_result_from_sub_agent") is None: # Ha az LLM pl. üres választ adott
             # Ezt a finalize_excel_processing kezeli, de itt is jelezhetjük.
             # A state-et itt nem módosítjuk, csak a router dönt.
             pass

        return "finalize"

workflow.add_conditional_edges(
    "excel_llm_planner_coder", # Forrás: az Excel LLM tervező node
    route_after_excel_planner, # A fenti új router függvény
    {
        "execute_code": "execute_excel_python_code",    # Ha kódot kell futtatni
        "finalize": "finalize_excel_processing"        # Ha a válasz megvan, vagy hiba történt a tervezőben/kódkinyerésnél
    }
)

workflow.add_conditional_edges(
    "execute_excel_python_code",  # A Python kód futtatása az Excel al-ágban UTÁN
    decide_next_excel_step,       # Ez a függvény dönti el a következő lépést az Excel al-ágban
    {
        "route_to_excel_llm_planner": "excel_llm_planner_coder", # Vissza az Excel LLM-hez újratervezni/következő lépést generálni
        "route_to_finalize_excel": "finalize_excel_processing"  # Kilépés az Excel al-ágból és eredmény összesítése
    }
)


# Az Excel al-ág vége visszacsatlakozik a fő gráfba:
# A `finalize_excel_processing` node után a fő agent `plan_next_step` node-ja következik,
# hogy feldolgozza az Excel al-ág eredményét (ami az `intermediate_steps`-be került).
workflow.add_edge("finalize_excel_processing", "plan_next_step")
workflow.add_edge("call_external_search_node", "plan_next_step")

workflow.add_edge("prepare_search_sub_graph", "search_query_refiner_or_planner_llm")

workflow.add_conditional_edges(
    "search_query_refiner_or_planner_llm", # Forrás: a keresési LLM tervező node
    decide_next_search_step,               # Ez a függvény dönti el a következő lépést az al-ágban
    {
        #"route_to_execute_external_search": "execute_native_search_node", # Ha új query-t kell futtatni
        #"route_to_search_llm_planner": "search_query_refiner_or_planner_llm", # Vissza az LLM-hez (pl. elemzésre, vagy ha hiba volt a query generálásban)
        #"route_to_finalize_search": "finalize_search_sub_graph"      # Ha a válasz kész, vagy max iteráció, vagy kritikus hiba

        "execute_native_search_node": "execute_native_search_node",
        "search_query_refiner_or_planner_llm": "search_query_refiner_or_planner_llm",
        "finalize_search_sub_graph": "finalize_search_sub_graph"
    }
)

# A keresés végrehajtása után vissza az LLM tervezőhöz/elemzőhöz
workflow.add_edge("execute_native_search_node", "search_query_refiner_or_planner_llm")

# Kilépés az al-ágból: a finalize_search_sub_graph után visszatérünk a fő ág tervezőjéhez
workflow.add_edge("finalize_search_sub_graph", "plan_next_step")

# --- Fő router (`should_continue_or_finish`) élei ---

# Az `execute_tool` (normál Pydantic toolok) után is a `should_continue_or_finish` dönt
workflow.add_conditional_edges(
    "plan_next_step",
    should_continue_or_finish,
    {
        "prepare_search_sub_graph": "prepare_search_sub_graph", # Ha a fő LLM a keresési al-ágat indítja
        "call_external_search_node": "call_external_search_node", # Ha a fő LLM a KÖZVETLEN külső szkript hívást kéri (ezt a verziót most nem használjuk, ha a sub-graph van)
        "prepare_excel_processing": "prepare_excel_processing",
        "execute_tool": "execute_tool", # Normál Pydantic toolokhoz
        "plan_next_step": "plan_next_step", # Újratervezés
        END: END
    }
)

workflow.add_conditional_edges(
    "execute_tool",
    should_continue_or_finish,
    {
        "plan_next_step": "plan_next_step",
        # A többi él itt valószínűleg nem szükséges, mert az execute_tool után
        # a should_continue_or_finish általában plan_next_step-re vagy END-re vezet.
        "prepare_search_sub_graph": "prepare_search_sub_graph", # Elméletileg nem innen indul
        "call_external_search_node": "call_external_search_node", # Elméletileg nem innen indul
        "prepare_excel_processing": "prepare_excel_processing", # Elméletileg nem innen indul
        "execute_tool": "execute_tool", # Csak ha egy tool egy másik Pydantic toolt hívna (nagyon ritka)
        END: END
    }
)

#workflow.add_edge("call_external_search_node", "plan_next_step")
# Vagy ha a should_continue_or_finish-t akarod használni itt is:
# workflow.add_conditional_edges(
#     "call_external_search_node",
#     should_continue_or_finish,
#     {
#         "plan_next_step": "plan_next_step", # Leggyakoribb
#         END: END # Ha a keresés után azonnal vége lenne (nem valószínű)
#     }
# )
# A sima add_edge("call_external_search_node", "plan_next_step") egyszerűbb és valószínűleg elég.



app = workflow.compile()
"""
# --- Gráf Vizualizálása (opcionális, de hasznos) ---
# Most már az 'app' (CompiledGraph) objektumot használjuk
try:
    # Telepítsd a szükséges csomagokat, ha még nem tetted meg:
    # pip install pygraphviz matplotlib
    # Vagy a mermaid-cli-t a mermaid png-hez: npm install -g @mermaid-js/mermaid-cli

    print("\nAttempting to generate workflow graph PNG...")
    # Az 'app' objektumnak van get_graph() metódusa
    graph_viz_object = app.get_graph()  # Itt hívjuk a get_graph()-ot

    # Kép mentése PNG-be
    # Előfordulhat, hogy a draw_mermaid_png() argumentumokat is vár, pl. output_file_path
    # Vagy lehet, hogy közvetlenül a graph_viz_object-en kell hívni.
    # Nézzük meg a LangGraph dokumentációját a pontos metódusnévre és használatra.
    # Egy gyakori mód:

    # Próbáljuk meg a Mermaid PNG-t, ha a pygraphviz problémás
    # Ehhez a `draw_mermaid_png` metódusnak léteznie kell a graph_viz_object-en.
    if hasattr(graph_viz_object, "draw_mermaid_png"):
        try:
            graph_png_bytes = graph_viz_object.draw_mermaid_png()
            if graph_png_bytes:
                with open("agent_workflow_graph.png", "wb") as f:
                    f.write(graph_png_bytes)
                print("✅ Workflow graph saved to agent_workflow_graph.png (using Mermaid)\n")
            else:
                print("ℹ️  draw_mermaid_png() returned None. Attempting ASCII.")
                graph_viz_object.print_ascii()
        except Exception as e_mermaid:
            print(f"ℹ️  Could not generate PNG with draw_mermaid_png(): {e_mermaid}. Attempting ASCII.")
            graph_viz_object.print_ascii()
    elif hasattr(graph_viz_object, "draw_png"):  # Régebbi LangGraph verziókhoz
        try:
            graph_viz_object.draw_png("agent_workflow_graph.png")
            print("✅ Workflow graph saved to agent_workflow_graph.png (using draw_png)\n")
        except Exception as e_draw_png:
            print(f"ℹ️  Could not generate PNG with draw_png(): {e_draw_png}. Attempting ASCII.")
            graph_viz_object.print_ascii()
    else:
        print("ℹ️  No PNG drawing method found (draw_mermaid_png or draw_png). Printing ASCII graph.")
        graph_viz_object.print_ascii()

except ImportError as e_import:
    print(f"\nℹ️  Graph visualization libraries might be missing (e.g., pygraphviz, matplotlib): {e_import}")
    print("   Attempting ASCII representation instead.")
    try:
        if 'app' in locals() and hasattr(app, 'get_graph'):  # Biztosítjuk, hogy az app létezik
            app.get_graph().print_ascii()
            print("--- End ASCII Workflow Graph ---\n")
        else:
            print("   Could not access graph for ASCII print (app or get_graph not available).")
    except Exception as e_ascii:
        print(f"   Failed to print ASCII graph as well: {e_ascii}")
except AttributeError as ae:  # Ha az app.get_graph() maga a hiba
    print(f"\nℹ️  AttributeError during graph visualization (likely app.get_graph()): {ae}")
    print("   This might indicate an issue with the LangGraph version or the compiled app object.")
    print("   Skipping graph visualization.")
except Exception as e_graph:
    print(f"\nℹ️  An unexpected error occurred during graph visualization: {e_graph}")
    print("   Skipping graph visualization.")

#-----Node-ok listája-----


print("--- Nodes in the Workflow ---")
print("-----------------------------")
for node_name, node_runnable in workflow.nodes.items():
    print(f"Node Name: {node_name}, Runnable: {node_runnable}")

#-----élek listája-----

print("\n--- Edges in the Compiled Graph ---")
print("-----------------------------")
if hasattr(app, 'get_graph') and hasattr(app.get_graph(), 'edges'):
    for edge in app.get_graph().edges:
        # Az 'edge' objektum formátuma LangGraph verziótól függhet.
        # Tipikusan tartalmazza a forrást (source), célt (target), és adatokat (data).
        source_node = getattr(edge, 'source', 'N/A')
        target_node = getattr(edge, 'target', 'N/A')
        edge_data = getattr(edge, 'data', {})  # Az adat tartalmazhatja a feltételt, ha feltételes él

        condition_info = ""
        # A feltételes élekhez tartozó feltételek a 'data' alatt lehetnek,
        # vagy a 'key' attribútumban, ha a conditional_edges egy dict-et adott vissza.
        # Ez a rész LangGraph verzióspecifikus lehet.
        # A `draw_mermaid_png` által generált Mermaid kód is jó forrás lehet az élek megértéséhez.
        # Egy egyszerűbb módja a feltételes élek megértésének a gráf vizuális vizsgálata.

        # Próbáljuk meg a 'data' attribútumot vizsgálni, hátha ott van a feltétel
        if isinstance(edge_data, dict):
            condition_label = edge_data.get('label',
                                            edge_data.get('key'))  # A 'label' vagy 'key' tartalmazhatja a feltételt
            if condition_label and condition_label != "__start__":  # __start__ egy speciális él
                condition_info = f" (Condition/Key: {condition_label})"

        print(f"Edge: From '{source_node}' To '{target_node}'{condition_info}")

else:
    print("Could not retrieve edge information directly. Try visualizing the graph.")

# A feltételes élek pontosabb listázása nehezebb, mert a router függvény logikájában vannak.
# A `workflow.branches` tartalmazhat információt a feltételes elágazásokról.
print("\n--- Conditional Edge Sources (Branches) ---")
print("-----------------------------")
if hasattr(workflow, 'branches'):
    for source_node, branches in workflow.branches.items():
        print(f"Branching from Node: {source_node}")
        for branch_key, target_node in branches.items():
            # A `branch_key` itt a feltétel neve (a router függvény visszatérési értéke)
            # A `target_node` pedig a cél node neve
            # A tényleges router függvényt (pl. should_continue_or_finish) külön kell megnézni a logika megértéséhez.
            print(f"  Condition '{branch_key}' -> To Node '{target_node}'")
else:
    print("Branch information not directly available via workflow.branches.")

#-----Toolok listákja-----


print("\n--- Registered LangChain Tools (from global list) ---")
print("-----------------------------")
if 'tools' in globals() and isinstance(tools, list):
    for tool_instance in tools:
        if hasattr(tool_instance, 'name') and hasattr(tool_instance, 'description'):
            print(f"Tool Name: {tool_instance.name}, Description: {tool_instance.description}")
            if hasattr(tool_instance, 'args_schema') and tool_instance.args_schema:
                # Kiírathatjuk az args_schema mezőit is, ha Pydantic modell
                try:
                    print(f"  Args Schema Fields: {list(tool_instance.args_schema.__fields__.keys())}")
                except AttributeError:
                    print(f"  Args Schema: (Not a Pydantic v1/v2 model with __fields__ or not inspectable easily)")
            else:
                print("  Args Schema: Not defined or N/A")
        else:
            print(f"Tool: {tool_instance} (Name/Description not found)")
else:
    print("Global 'tools' list not found or not a list.")

#-----Függvények listája-----

print("\n--- Functions Used as Nodes ---")
print("-----------------------------")
node_functions = set()
if hasattr(workflow, 'nodes'):
    for node_name, node_object in workflow.nodes.items():
        # A node_object egy belső LangGraph reprezentációja a node-nak.
        # A futtatható függvényt a 'runnable' attribútumán keresztül érhetjük el,
        # vagy maga a node_object lehet az, ha egyszerűbb a struktúra.
        # A te kiíratásod alapján a node_object-nek van 'runnable' attribútuma.

        runnable_function = None
        if hasattr(node_object, 'runnable'):  # Ha a Node objektumnak van 'runnable' attribútuma
            runnable_function = node_object.runnable
        elif callable(node_object):  # Ha maga a node_object a függvény (ritkább StateGraph-nél)
            runnable_function = node_object

        if runnable_function and callable(runnable_function):
            function_name = getattr(runnable_function, '__name__', str(runnable_function))
            print(f"Node Name: {node_name} --- Runs Function: {function_name}")
        else:
            print(f"Node Name: {node_name} --- Runnable: {node_object} (Not a direct function or name not found)")
else:
    print("Could not access workflow.nodes")
"""

# --- Futtatás (példa) ---
if __name__ == "__main__":  # Fontos, hogy a fő futtató kód itt legyen

    #pass
    questions_file = Path("./gaia_level1_questions.json")
    parsed_questions = []
    if questions_file.exists():
        import json

        with open(questions_file, "r", encoding="utf-8") as f:
            parsed_questions = json.load(f)
    else:
        print(f"Warning: {questions_file} not found. No questions to process.")

    if parsed_questions:
        # Kérdés fájl nélkül (Mercedes Sosa)
        # question_to_test = next((q for q in parsed_questions if q["task_id"] == "8e867cd7-cff9-4e6c-867a-ff5ddc2550be"), None)

        # Kérdés fájllal (Python kód) - ehhez még kell a PythonInterpreterTool
        question_to_test = next((q for q in parsed_questions if q["task_id"] == "8e867cd7-cff9-4e6c-867a-ff5ddc2550be"), None)
        #question_to_test=""
        # Kérdés fájllal (Excel) - ehhez kell a SpreadsheetProcessorTool (vagy PythonInterpreter+pandas)
        # question_to_test = next((q for q in parsed_questions if q["task_id"] == "7bd855d8-463d-4ed5-93ca-5fe35145f733"), None)

        if question_to_test:

            print(f"\n--- TESTING AGENT ON QUESTION ---")
            print(f"Task ID: {question_to_test['task_id']}")
            print(f"Question: {question_to_test['question']}")

            config = {"recursion_limit": 25}
            initial_state = {
                "original_question": question_to_test["question"],
                "task_id": question_to_test["task_id"],
                # A többi állapotmezőt az 'initialize_agent' nullázza/beállítja
            }

            final_state_result = None
            print("\n--- AGENT STREAM STARTED ---")
            try:
                for event_update in app.stream(initial_state, config=config, stream_mode="values"):
                    # Az 'event_update' az AgentState aktuális állapota minden node lefutása után
                    # print(f"Current state after a step: {event_update}") # Nagyon részletes lenne
                    final_state_result = event_update  # Az utolsó esemény lesz a végső állapot

                print("\n--- AGENT EXECUTION FINISHED ---")
                if final_state_result:
                    #print("\nDEBUG: Tartalma a final_state_result['intermediate_steps']-nek a kiíratás előtt:")
                    #print(final_state_result.get('intermediate_steps'))  # Ezt nézd meg alaposan!
                    print(f"  Final Answer: {final_state_result.get('final_answer')}")
                    print(f"  File Path (if any): {final_state_result.get('file_path')}")
                    print(f"  Error (if any): {final_state_result.get('error_message')}")
                    #print(f"  Intermediate Steps ({len(final_state_result.get('intermediate_steps', []))}):")
                    #for i, step in enumerate(final_state_result.get('intermediate_steps', [])):
                    #    tool_name, tool_output = step
                    #    print(
                    #        f"    {i + 1}. Tool: {tool_name}, Output: {str(tool_output)[:300]}...")  # Rövidített kimenet
                else:
                    print("  No final state recorded from stream.")

            except Exception as e:
                print(f"An error occurred during agent execution stream: {e}")
                import traceback

                traceback.print_exc()
        else:
            print("Could not find the specified test question.")
    else:
        print("No parsed questions to test with.")