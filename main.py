from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from anthropic import Anthropic
import base64, httpx, json, os
from datetime import date

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

app = FastAPI()
client = Anthropic(api_key=ANTHROPIC_API_KEY)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SYSTEM_PROMPT = """Du är en nutritionsexpert. Analysera matinput och returnera ALLTID giltig JSON.
Format:
{
  "items": [
    {
      "name": "Livsmedel",
      "amount_g": 100,
      "calories": 150,
      "protein_g": 10,
      "carbs_g": 15,
      "fat_g": 5,
      "confidence": "high|medium|low",
      "note": "valfri kommentar vid osäkerhet"
    }
  ],
  "total_calories": 150,
  "total_protein_g": 10,
  "total_carbs_g": 15,
  "total_fat_g": 5
}
Svara ENDAST med JSON, ingen annan text."""

def parse_nutrition(content: str) -> dict:
    clean = content.strip().replace("```json", "").replace("```", "")
    return json.loads(clean)

@app.post("/log/text")
async def log_text(text: str = Form(...)):
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}]
    )
    return parse_nutrition(resp.content[0].text)

@app.post("/log/audio")
async def log_audio(audio: UploadFile = File(...)):
    # Transkribera med Whisper
    audio_bytes = await audio.read()
    async with httpx.AsyncClient() as http:
        whisper_resp = await http.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": (audio.filename, audio_bytes, audio.content_type)},
            data={"model": "whisper-1", "language": "sv"}
        )
    whisper_data = whisper_resp.json()
    if "text" not in whisper_data:
        return {"error": "Whisper fel", "details": whisper_data}
    transcript = whisper_data["text"]
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": transcript}]
    )
    result = parse_nutrition(resp.content[0].text)
    result["transcript"] = transcript
    return result

@app.post("/log/photo")
async def log_photo(photo: UploadFile = File(...)):
    img_bytes = await photo.read()
    b64 = base64.standard_b64encode(img_bytes).decode()
    resp = client.messages.create(
        model="claude-sonnet-4-5",  # Vision kräver Sonnet
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": "Analysera maten på bilden och uppskatta nutritionsvärden."}
            ]
        }]
    )
    return parse_nutrition(resp.content[0].text)

@app.post("/log/barcode")
async def log_barcode(barcode: str = Form(...)):
    # Slå upp i Open Food Facts
    async with httpx.AsyncClient() as http:
        off_resp = await http.get(f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json")
    data = off_resp.json()
    if data.get("status") == 1:
        n = data["product"].get("nutriments", {})
        name = data["product"].get("product_name", "Okänd produkt")
        per100 = {
            "name": name,
            "amount_g": 100,
            "calories": n.get("energy-kcal_100g", 0),
            "protein_g": n.get("proteins_100g", 0),
            "carbs_g": n.get("carbohydrates_100g", 0),
            "fat_g": n.get("fat_100g", 0),
            "confidence": "high",
            "note": "Verifierad data från Open Food Facts"
        }
        return {"items": [per100], "source": "openfoodfacts", **{k: per100[k] for k in ["calories","protein_g","carbs_g","fat_g"]},
                "total_calories": per100["calories"], "total_protein_g": per100["protein_g"],
                "total_carbs_g": per100["carbs_g"], "total_fat_g": per100["fat_g"]}
    # Fallback: fråga LLM med streckkodsnummer
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Streckkod {barcode} hittades inte i databas. Ge ett generiskt svar om okänd produkt."}]
    )
    return parse_nutrition(resp.content[0].text)

# Health check
@app.get("/")
def root():
    return {"status": "ok", "date": str(date.today())}
