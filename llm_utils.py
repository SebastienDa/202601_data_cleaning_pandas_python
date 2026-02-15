import google.generativeai as genai
import pandas as pd
import json
import io
from pydantic import BaseModel, Field
from typing import List

# --- Pydantic Models for Structured Output ---

class DataIssue(BaseModel):
    column: str = Field(..., description="Nom de la colonne concernée")
    issue_type: str = Field(..., description="Type de problème (ex: 'Format', 'Duplicata', 'Valeur manquante', 'Incohérence')")
    description: str = Field(..., description="Explication courte et technique du problème")
    suggested_action: str = Field(..., description="Action Python suggérée (ex: 'pd.to_datetime', 'drop_duplicates')")

class DataAudit(BaseModel):
    issues: List[DataIssue]

class CodeOutput(BaseModel):
    python_script: str = Field(..., description="Le script Python complet et exécutable pour nettoyer les données, sans blocs markdown (```python).")

# --- Core Functions ---

def configure_genai(api_key: str):
    """Calibre le client Gemini avec la clé API fournie."""
    genai.configure(api_key=api_key)

def analyze_dataframe(df: pd.DataFrame, api_key: str) -> List[dict]:
    """
    Analyse le dataframe et retourne une liste de problèmes structurés.
    """
    configure_genai(api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')

    # Capture des infos techniques
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    head_str = df.head(50).to_markdown()

    # Prompt "Expert Impitoyable"
    prompt = f"""
    Tu es un Expert Data Engineer Senior spécialisé en Pandas.
    Audite ce dataset pour produire un plan de nettoyage détaillé.

    DATA HEAD:
    {head_str}

    DATA INFO:
    {info_str}

    Ta mission :
    1. Détecte les anomalies de TYPES (Object au lieu de Date/Float).
    2. Détecte les anomalies de FORMAT (Gère les formats de date mixtes, les prix avec symboles).
    3. Détecte les anomalies SÉMANTIQUES (Doublons d'ID, quantités négatives, fautes de frappe catégories).
    4. Propose des solutions robustes (errors='coerce', regex).

    IMPORTANT : Remplis bien TOUS les champs, y compris "description".
    Retourne le résultat strictement au format JSON défini.
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json", "response_schema": DataAudit}
        )
        # Conversion Pydantic -> Dict pour Streamlit
        audit_result = json.loads(response.text)
        
        # DEBUG LOG
        if audit_result['issues']:
            print("DEBUG KEYS:", audit_result['issues'][0].keys())
            
        return audit_result['issues']
    except Exception as e:
        print(f"Erreur Analyse LLM : {e}")
        return []

def generate_cleaning_code(df_sample: pd.DataFrame, selected_issues: List[dict], api_key: str) -> str:
    """
    Génère le code Python pour appliquer les corrections sélectionnées.
    """
    if not selected_issues:
        return "print('Aucune action sélectionnée.')"

    configure_genai(api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    cols_context = df_sample.dtypes.to_dict()

    prompt = f"""
    Tu es un développeur Python Senior expert en Pandas.
    Génère un script Python pour nettoyer un dataframe nommé 'df' en appliquant UNIQUEMENT ces actions validées :
    {json.dumps(selected_issues, indent=2)}

    CONTEXTE COLONNES:
    {cols_context}

    CONTRAINTES STRICTES :
    1. Le code doit agir directement sur la variable 'df'.
    2. Utilise des méthodes ROBUSTES (pd.to_numeric avec errors='coerce', regex pour nettoyer les strings).
    3. Gère les dates mixtes si mentionné.
    4. Ne retourne QUE du code Python valide, sans markdown.
    5. LIBRARIES DISPONIBLES : pandas (pd), numpy (np), re. N'utilise PAS d'autres librairies.
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json", "response_schema": CodeOutput}
        )
        return json.loads(response.text)['python_script']
    except Exception as e:
        return f"# Erreur Génération Code : {e}"
