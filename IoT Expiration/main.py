# === PART 1: IMPORTS ===
import cv2
import easyocr
import re
import datetime
import uvicorn
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Optional, List

# === PART 2: CONFIGURATION & INITIALIZATION ===

# --- Google AI API Key Configuration ---
GOOGLE_API_KEY = "AIzaSyCSGa3UQ7WGmeSOW4jHAIWRwgv3kXQNi34" # Your key should be here
genai.configure(api_key=GOOGLE_API_KEY)

# --- FastAPI App ---
app = FastAPI(
    title="Anti-Waste Smart Scanner API (Final Logic)",
    description="Uses your preferred text-based logic to find the expiry date."
)

# --- OCR Reader ---
ocr_reader = easyocr.Reader(['fr', 'en'])

# --- Expiry Keywords ---
# Standard, case-insensitive keywords
EXPIRY_KEYWORDS = [
    'exp', 'expiry', 'use by', 'best before', 'expires',
    'exp le', 'a consommer avant le', 'a cons de pref avant'
]

# --- New: Case-sensitive keywords ---
CASE_SENSITIVE_KEYWORDS = ["Ut. Av:"]

# --- New: French Month Abbreviation Map ---
FRENCH_MONTH_MAP = {
    'jan': 1, 'fev': 2, 'mar': 3, 'avr': 4, 'mai': 5, 'jui': 6,
    'jul': 7, 'aou': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

# === PART 3: HELPER FUNCTIONS ===

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    # This function is unchanged
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)
    return contrast_enhanced

# ------------------------------------------------------------------
# vv THIS IS THE ONLY FUNCTION THAT HAS BEEN CHANGED vv
# ------------------------------------------------------------------
def find_expiry_date(ocr_text_blocks: List[str]) -> Optional[str]:
    """
    Finds expiry date using text-based "closest" logic.
    Now supports "MMM YY" format and new case-sensitive keywords.
    """
    # --- Helper function to convert 2-digit years ---
    def convert_yy_to_yyyy(y: int) -> int:
        return 2000 + y if y <= 50 else 1900 + y

    # Get original text for case-sensitive searches and lowercase for others.
    full_text_orig = " ".join(ocr_text_blocks)
    full_text_lower = full_text_orig.lower()

    # --- 1. Find all keyword positions ---
    keyword_positions = []
    # Find positions of standard lowercase keywords.
    for keyword in EXPIRY_KEYWORDS:
        for match in re.finditer(re.escape(keyword), full_text_lower):
            keyword_positions.append(match.start())
            
    # Find position of uppercase 'E' ONLY.
    for match in re.finditer(r'\bE\b', full_text_orig):
        keyword_positions.append(match.start())

    # Find position of new case-sensitive keywords like "Ut Av:".
    for keyword in CASE_SENSITIVE_KEYWORDS:
        for match in re.finditer(re.escape(keyword), full_text_orig):
            keyword_positions.append(match.start())

    # --- 2. Find all date positions and parse them ---
    dates_found = []
    patterns = {
        'dmy': [
            r'\b(0[1-9]|[12][0-9]|3[01])[-/](0[1-9]|1[0-2])[-/](20\d{2})\b', # DD/MM/YYYY
            r'\b(0[1-9]|[12][0-9]|3[01])(0[1-9]|1[0-2])(20\d{2})\b',       # DDMMYYYY
            r'\b(0[1-9]|[12][0-9]|3[01])[-/](0[1-9]|1[0-2])[-/](\d{2})\b',    # DD/MM/YY
            r'\b(0[1-9]|[12][0-9]|3[01])(0[1-9]|1[0-2])(\d{2})\b'           # DDMMYY
        ],
        'my': [
            r'\b(0[1-9]|1[0-2])[-/](\d{2})\b' # MM/YY pattern
        ],
        'mmm_yy': [ # New MMM YY pattern
            r'\b(jan|fev|mar|avr|mai|jui|jul|aou|sep|oct|nov|dec)[. ]*(\d{2})\b'
        ]
    }

    # Process standard Day-Month-Year formats
    for pattern in patterns['dmy']:
        for match in re.finditer(pattern, full_text_lower):
            d, m, y_str = match.groups(); year = int(y_str) if len(y_str) == 4 else convert_yy_to_yyyy(int(y_str))
            try: dates_found.append({'date': datetime.date(year, int(m), int(d)), 'pos': match.start()}); continue
            except ValueError: continue

    # Process MM/YY format
    for pattern in patterns['my']:
        for match in re.finditer(pattern, full_text_lower):
            m, y_str = match.groups(); year = convert_yy_to_yyyy(int(y_str))
            try: dates_found.append({'date': datetime.date(year, int(m), 1), 'pos': match.start()}); continue
            except ValueError: continue
            
    # Process new MMM YY format
    for pattern in patterns['mmm_yy']:
        for match in re.finditer(pattern, full_text_lower):
            month_abbr, year_str = match.groups(); month_num = FRENCH_MONTH_MAP.get(month_abbr)
            if month_num:
                year = convert_yy_to_yyyy(int(year_str))
                try: dates_found.append({'date': datetime.date(year, month_num, 1), 'pos': match.start()}); continue
                except ValueError: continue

    # --- 3. Find the best date based on what was found ---
    if not dates_found: return None

    if keyword_positions:
        closest_date = None; min_distance = float('inf')
        for key_pos in keyword_positions:
            for date_info in dates_found:
                distance = abs(key_pos - date_info['pos'])
                if distance < min_distance:
                    min_distance = distance; closest_date = date_info['date']
        if closest_date: return closest_date.strftime("%d/%m/%Y")

    return max([info['date'] for info in dates_found]).strftime("%d/%m/%Y")
# ------------------------------------------------------------------
# ^^ END OF THE CHANGED FUNCTION ^^
# ------------------------------------------------------------------

def get_product_status(expiry_date_str: str) -> dict:
    # This function is unchanged
    expiry_date = datetime.datetime.strptime(expiry_date_str, "%d/%m/%Y").date()
    today = datetime.date.today()
    three_days_from_now = today + datetime.timedelta(days=3)
    if expiry_date < today: return {"status": "Périmé", "message": "Product is expired."}
    if today <= expiry_date <= three_days_from_now: return {"status": "Alerte Gaspillage", "message": "Product is nearing its expiry date."}
    return {"status": "Valide", "message": "Product is valid."}

def get_llm_recommendation(status: str, product_name: str = "this product") -> Optional[str]:
    # This function is unchanged
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
        if status in ["Alerte Gaspillage", "Périmé"]: return "Recommendation engine disabled: No Google API Key provided."
        return None
        
    model = genai.GenerativeModel('models/gemini-flash-latest')

    prompt = ""
    if status == "Alerte Gaspillage": prompt = f"Act as a culinary assistant. {product_name.capitalize()} is about to expire. Suggest a simple and quick recipe to use it immediately. Be concise."
    elif status == "Périmé": prompt = f"Act as a health and safety advisor. {product_name.capitalize()} is expired. Briefly indicate the potential health risks of consuming it and the proper way to dispose of the packaging. Be concise."
    else: return None
    try: return model.generate_content(prompt).text
    except Exception as e: return f"Error generating recommendation: {e}"


# === PART 4: API ENDPOINT ===
@app.post("/scan")
async def scan_product_image(file: UploadFile = File(...)):
    # This section is unchanged and uses your preferred OCR method.
    if not file.content_type.startswith("image/"): raise HTTPException(status_code=400, detail="File provided is not an image.")
    image_bytes = await file.read()
    preprocessed_image = preprocess_image(image_bytes)
    
    ocr_results = ocr_reader.readtext(preprocessed_image, detail=0, paragraph=True)
    
    if not ocr_results: raise HTTPException(status_code=404, detail="No text could be detected in the image.")

    extracted_date = find_expiry_date(ocr_results)
    
    if not extracted_date: raise HTTPException(status_code=404, detail=f"No valid expiry date was found. Detected text: '{' '.join(ocr_results)}'")
        
    status_info = get_product_status(extracted_date)
    recommendation = get_llm_recommendation(status_info["status"])

    return {
        "extracted_date": extracted_date,
        "status": status_info["status"],
        "message": status_info["message"],
        "recommendation": recommendation,
        "full_text_detected": " ".join(ocr_results)
    }
