"""
health_nudges.py
────────────────────────────────────────────────────────────────────────────
IRIS Patient Portal — Personalized Health Nudges Module

Generates weekly, LLM-powered, personalized health tips based on lab results.
Designed to plug directly into the existing Flask app and RAG pipeline.

Key Design Decisions:
  - Uses the same lab JSON schema as lab_results.json (test_name / status: Low/High)
  - Strips all patient PII before sending to LLM (HIPAA-mindful)
  - Returns structured JSON nudges (3 categories: Diet, Lifestyle, See Your Doctor)
  - Includes APScheduler hook for weekly digest emails

Flask integration:
  Add to app.py:
    from health_nudges import generate_health_nudges

    @app.route('/api/nudges', methods=['POST'])
    def nudges():
        lab_data = request.json or load_sample_labs()
        result = generate_health_nudges(lab_data)
        return jsonify(result)

Author : IRIS Digi Corp
Version: 1.0.0
"""

import os
import json
import logging
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load .env so OPENAI_API_KEY is available whether this module
# is imported by app.py or run directly as a standalone script.
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def _get_client() -> OpenAI:
    """Lazy client init — ensures .env is loaded before reading the key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Add it to your .env file:\n"
            "  OPENAI_API_KEY=sk-..."
        )
    return OpenAI(api_key=api_key)

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a compassionate women's health assistant for the IRIS platform.
Your role is to turn abnormal lab results into warm, practical, actionable 
weekly health nudges — like advice from a knowledgeable friend, not a clinical report.

Rules:
- Plain language only. No jargon.
- Never diagnose. Use "may suggest" or "could be related to."
- Always recommend following up with their doctor.
- Be encouraging. These are nudges, not warnings.
- Output ONLY valid JSON — no markdown, no backticks, no extra text.
"""

USER_PROMPT_TEMPLATE = """
Here is a de-identified summary of a patient's abnormal lab results:

{abnormal_summary}

Generate a personalized weekly health nudge report.

Return valid JSON in exactly this structure:
{{
  "headline": "One warm, personalized sentence summarizing what these results mean overall",
  "focus_condition": "Plain-English name of the primary condition these results suggest",
  "nudges": {{
    "diet": [
      {{"tip": "...", "reason": "...", "try_this": "..."}},
      {{"tip": "...", "reason": "...", "try_this": "..."}},
      {{"tip": "...", "reason": "...", "try_this": "..."}}
    ],
    "lifestyle": [
      {{"tip": "...", "reason": "...", "try_this": "..."}},
      {{"tip": "...", "reason": "...", "try_this": "..."}},
      {{"tip": "...", "reason": "...", "try_this": "..."}}
    ],
    "talk_to_your_doctor": [
      {{"bring_up": "...", "why": "..."}},
      {{"bring_up": "...", "why": "..."}},
      {{"bring_up": "...", "why": "..."}}
    ]
  }},
  "encouragement": "One closing sentence that motivates the patient to take small steps today"
}}
"""


# ── Core Logic ────────────────────────────────────────────────────────────────

def build_abnormal_summary(lab_data: dict) -> str:
    """
    Extract only abnormal results and format them for the LLM prompt.
    Note: We intentionally exclude patient name, DOB, and MRN — only
    test values are sent (minimum necessary, HIPAA-mindful design).
    """
    abnormal_lines = []
    for lab in lab_data.get("lab_results", []):
        if lab.get("status") in ("Low", "High"):
            line = (
                f"- {lab['test_name']}: {lab['value']} {lab['unit']} "
                f"(Reference: {lab['reference_range']}, Flag: {lab['flag']})"
            )
            abnormal_lines.append(line)

    if not abnormal_lines:
        return "All results within normal range."

    return "\n".join(abnormal_lines)


def detect_primary_condition(lab_data: dict) -> str:
    """
    Simple heuristic to detect the dominant condition pattern.
    Keeps the nudges focused rather than scattering across all issues.
    In production, this would use a proper condition classifier.
    """
    test_names = {lab["test_name"] for lab in lab_data.get("lab_results", [])
                  if lab.get("status") in ("Low", "High")}

    iron_markers = {"Serum Iron", "Ferritin", "TIBC", "Transferrin Saturation", "MCV"}
    lipid_markers = {"Total Cholesterol", "LDL Cholesterol", "Triglycerides"}

    iron_hits = len(test_names & iron_markers)
    lipid_hits = len(test_names & lipid_markers)

    if iron_hits >= 2:
        return "Iron Deficiency Anemia"
    elif lipid_hits >= 2:
        return "Elevated Lipid Panel"
    elif "Glucose (Fasting)" in test_names:
        return "Elevated Blood Sugar"
    elif "Vitamin D" in test_names:
        return "Vitamin D Deficiency"
    else:
        return "Multiple Abnormal Values"


def generate_health_nudges(lab_data: dict) -> dict:
    """
    Main entry point.

    Takes the same lab_data dict that load_sample_labs() returns.
    Returns a structured nudge payload ready for the Flask response or email digest.

    Args:
        lab_data (dict): Lab results in the existing app format.

    Returns:
        dict: {
            generated_at, patient_id, focus_condition, nudge_content, disclaimer
        }
    """
    abnormal_summary = build_abnormal_summary(lab_data)
    primary_condition = detect_primary_condition(lab_data)

    logger.info(f"Generating nudges | Condition: {primary_condition} | "
                f"Abnormal tests: {abnormal_summary.count(chr(10)) + 1}")

    prompt = USER_PROMPT_TEMPLATE.format(abnormal_summary=abnormal_summary)

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.4,
            max_tokens=1200,
        )

        raw = response.choices[0].message.content.strip()
        nudge_content = json.loads(raw)
        logger.info("Nudge generation successful.")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        nudge_content = {
            "headline": "Your results are ready — let's look at some ways to support your health.",
            "focus_condition": primary_condition,
            "nudges": {
                "diet": [{"tip": "Eat more iron-rich foods", "reason": "Supports red blood cell production",
                          "try_this": "Add lentils or spinach to one meal today"}],
                "lifestyle": [{"tip": "Get adequate sleep", "reason": "Supports recovery",
                               "try_this": "Aim for 7-9 hours tonight"}],
                "talk_to_your_doctor": [{"bring_up": "Your abnormal lab values",
                                         "why": "To get a personalized treatment plan"}]
            },
            "encouragement": "Small steps every day add up to big changes in how you feel."
        }

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "patient_id": lab_data.get("patient", {}).get("mrn", "unknown"),
        "focus_condition": primary_condition,
        "nudge_content": nudge_content,
        "disclaimer": (
            "These nudges are for informational purposes only and do not constitute "
            "medical advice. Always consult your healthcare provider before making "
            "changes to your diet, supplements, or medications."
        )
    }


# ── APScheduler Hook ──────────────────────────────────────────────────────────
"""
Weekly email digest — add this to your scheduler setup:

from apscheduler.schedulers.background import BackgroundScheduler
from health_nudges import generate_health_nudges
import json

def send_weekly_nudge_digest():
    '''Runs every Monday at 8am. In production, loops over active users from DB.'''
    with open('sample_data/lab_results.json') as f:
        lab_data = json.load(f)

    payload = generate_health_nudges(lab_data)

    # send_nudge_email(
    #     to=lab_data['patient']['email'],
    #     subject=f"Your Weekly Health Nudge: {payload['focus_condition']}",
    #     nudge_payload=payload
    # )

    print(f"[Nudge Sent] {payload['focus_condition']} — {payload['generated_at']}")

scheduler = BackgroundScheduler()
scheduler.add_job(send_weekly_nudge_digest, 'cron', day_of_week='mon', hour=8)
scheduler.start()
"""


# ── Standalone Demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    lab_path = os.path.join("sample_data", "lab_results.json")
    if not os.path.exists(lab_path):
        print(f"[ERROR] Could not find {lab_path}")
        sys.exit(1)

    with open(lab_path) as f:
        lab_data = json.load(f)

    print("\n" + "=" * 60)
    print("  IRIS Patient Portal — Weekly Health Nudges")
    print("=" * 60)

    result = generate_health_nudges(lab_data)
    nc = result["nudge_content"]

    print(f"\n{nc['headline']}")
    print(f"   Focus: {result['focus_condition']}\n")

    for section_key, label in [("diet", "DIET"), ("lifestyle", "LIFESTYLE"),
                                 ("talk_to_your_doctor", "TALK TO YOUR DOCTOR")]:
        print(f"-- {label} --")
        for i, item in enumerate(nc["nudges"][section_key], 1):
            if section_key == "talk_to_your_doctor":
                print(f"  {i}. Bring up: {item['bring_up']}")
                print(f"     Why: {item['why']}")
            else:
                print(f"  {i}. {item['tip']}")
                print(f"     Why: {item['reason']}")
                print(f"     Try: {item['try_this']}")
        print()

    print(f"{nc['encouragement']}\n")
    print(f"   Generated: {result['generated_at']}")
    print("=" * 60 + "\n")