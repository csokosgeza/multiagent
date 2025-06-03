# external_google_search_EXACT.py

import os
import sys
# Az alapkódod ezt használta:
from google import genai  # Ez a `google-generativeai` csomagra kellene, hogy utaljon
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch


# Szükséges lehet a load_dotenv, ha a szkript maga kezeli a kulcsot
from dotenv import load_dotenv
load_dotenv()

def search_google_exact_original(query_string: str,
                                 model_id_str: str = "gemini-2.5-flash-preview-05-20",  # Modell az alapkódodból
                                 # Az alapkódod nem specifikált temperature-t stb. a configon kívül,
                                 # de a GenerateContentConfig-ban lehetnek.
                                 ):
    """
    Megpróbálja végrehajtani a Google keresést PONTOSAN a felhasználó
    által adott, működő kód szintaxisával.
    FIGYELEM: Ez hibát dobhat, ha a telepített SDK verzió nem támogatja
    a `client.models.generate_content` szintaxist.
    """
    print(f"--- Attempting Search with EXACT Original Snippet Logic ---")
    print(f"Query: '{query_string}'")
    print(f"Model: '{model_id_str}'")

    try:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("ERROR_API_KEY_NOT_FOUND")
            return

        # Az alapkódodban nem volt explicit genai.configure(),
        # feltételezve, hogy a Client() ezt kezeli, vagy globálisan be volt állítva.
        # A biztonság kedvéért itt meghagyom, de ha az eredetiben nem volt, kivehető.
        try:
            genai.configure(api_key=api_key)
            print(f"genai SDK (version: {getattr(genai, '__version__', 'unknown')}) configured.")
        except Exception as e:
            # print(f"Note: genai.configure error (maybe already called): {e}")
            pass

        # Kliens példányosítása, ahogy az alapkódodban
        client = genai.Client()  # Ennek kell visszaadnia egy olyan klienst, aminek van .models
        print("genai.Client() called.")

        # Google Search eszköz létrehozása
        google_search_tool_sdk = Tool(
            google_search=GoogleSearch()
        )
        print(f"SDK Tool with GoogleSearch created: {google_search_tool_sdk}")

        # Generálási konfiguráció az OSZTÁLY PÉLDÁNYOSÍTÁSÁVAL
        # Az alapkódodban a response_modalities is itt volt.
        config_for_api = GenerateContentConfig(
            tools=[google_search_tool_sdk],
            response_modalities=["TEXT"]  # Ahogy az alapkódodban
            # temperature és candidate_count nem volt az alapkódod configjában,
            # de ha kellenek, itt hozzáadhatók, ha a GenerateContentConfig ismeri őket.
            # temperature=0.1,
            # candidate_count=1
        )
        print(f"SDK GenerateContentConfig instance created: {config_for_api}")

        # A `model` argumentum az alapkódodban csak a modell ID volt, nem a teljes "models/..." elérési út.
        # A `client.models.generate_content` lehet, hogy ezt így várta.

        # === EZ A KRITIKUS SOR AZ ALAPKÓDODBÓL ===
        # Ha ez AttributeError: 'Client' object has no attribute 'models' hibát dob,
        # akkor a `genai.Client()` a `google-generativeai` SDK-ban nem az, amit az alapkódod használt.
        response = client.models.generate_content(
            model=model_id_str,  # Az alapkódodban csak a modell ID volt
            contents=query_string,
            config=config_for_api
        )
        # === EDDIG ===
        print("client.models.generate_content call completed.")

        output_parts = []
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part_obj in response.candidates[0].content.parts:
                if hasattr(part_obj, 'text') and part_obj.text:
                    output_parts.append(part_obj.text)

        final_output = "\n".join(output_parts).strip()

        if final_output:
            print(final_output)
        else:
            print("ERROR_NO_TEXT_OUTPUT_FROM_SEARCH")
            if response.candidates and response.candidates[0].finish_reason:
                print(f"Finish Reason: {response.candidates[0].finish_reason}")


    except ImportError as ie:
        print(f"ERROR_IMPORT:{ie}")
    except AttributeError as ae:  # Ezt várjuk, ha a client.models nem létezik
        print(f"ERROR_ATTRIBUTE:{ae}")
    except Exception as e:
        print(f"ERROR_UNEXPECTED:{e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        # Az alapkódod "gemini-2.0-flash"-t használt
        model_to_use = "gemini-2.5-flash-preview-05-20"
        if len(sys.argv) > 2:
            model_to_use = sys.argv[2]  # Lehetővé teszi a modell felülbírálását
        search_google_exact_original(query, model_id_str=model_to_use)
    else:
        print("Usage: python external_google_search_EXACT.py \"<your query>\" [model_id]")
        # Példa hívás teszteléshez:
        #search_google_exact_original("Milyen nap van ma?")