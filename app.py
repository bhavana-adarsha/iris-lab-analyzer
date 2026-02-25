import json
import os
from flask import Flask, jsonify, render_template, request
from openai import OpenAI
from dotenv import load_dotenv
from health_nudges import generate_health_nudges

# Load environment variables
load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def load_sample_labs():
    with open('sample_data/lab_results.json', 'r') as f:
        return json.load(f)

def load_medical_knowledge():
    with open('sample_data/medical_knowledge.json', 'r') as f:
        return json.load(f)

def retrieve_relevant_knowledge(lab_results, knowledge_base):
    """RAG STEP 1: RETRIEVAL — retrieve relevant medical knowledge for each test"""
    relevant_knowledge = {}
    for lab in lab_results:
        test_name = lab['test_name']
        if test_name in knowledge_base['tests']:
            test_info = knowledge_base['tests'][test_name]
            relevant_info = {'description': test_info['description']}
            if lab['status'] == 'Low' and 'low_causes' in test_info:
                relevant_info['causes'] = test_info['low_causes']
                relevant_info['symptoms'] = test_info.get('low_symptoms', [])
                relevant_info['actions'] = test_info.get('recommended_actions_low', [])
            elif lab['status'] == 'High' and 'high_causes' in test_info:
                relevant_info['causes'] = test_info['high_causes']
                relevant_info['symptoms'] = test_info.get('high_symptoms', [])
                relevant_info['actions'] = test_info.get('recommended_actions_high', [])
            relevant_knowledge[test_name] = relevant_info
    return relevant_knowledge

# HIPAA Notes (production):
# - Azure OpenAI with BAA for HIPAA-compliant cloud
# - De-identify before LLM: strip name, DOB, MRN
# - TLS 1.3 + AES-256 encryption, audit logs, no PHI retention

def analyze_labs_with_gpt(lab_data):
    """FULL RAG PIPELINE: Retrieve → Augment → Generate"""
    knowledge_base = load_medical_knowledge()
    retrieved_knowledge = retrieve_relevant_knowledge(lab_data['lab_results'], knowledge_base)

    lab_summary = []
    abnormal_results = []
    for lab in lab_data['lab_results']:
        status_marker = f" [{lab['flag']}]" if lab['flag'] else ""
        lab_summary.append(f"- {lab['test_name']}: {lab['value']} {lab['unit']} (Normal: {lab['reference_range']}){status_marker}")
        if lab['status'] != 'Normal':
            abnormal_results.append(lab)

    knowledge_context = "\n\nMEDICAL KNOWLEDGE BASE (use this to provide accurate explanations):\n"
    for test_name, info in retrieved_knowledge.items():
        knowledge_context += f"\n{test_name}:\n- What it is: {info['description']}\n"
        if 'causes' in info:
            knowledge_context += f"- Common causes: {', '.join(info['causes'])}\n"
        if 'symptoms' in info:
            knowledge_context += f"- Symptoms: {', '.join(info['symptoms'])}\n"
        if 'actions' in info:
            knowledge_context += f"- Recommended actions: {', '.join(info['actions'])}\n"

    prompt = f"""You are a helpful health assistant analyzing lab results.

Patient Information:
- Name: {lab_data['patient']['name']}
- Test Date: {lab_data['test_date']}

Lab Results:
{chr(10).join(lab_summary)}

{knowledge_context}

Additional Clinical Notes: {lab_data.get('notes', 'None')}

Please provide a comprehensive analysis:

1. **Overall Health Summary** (2-3 sentences)
2. **Abnormal Results Explained** — for each: what it measures, possible causes, symptoms to watch for
3. **Actionable Recommendations** (5-7 steps: diet, lifestyle, follow-up tests, when to see a doctor)
4. **Key Discussion Points for Your Doctor**

GUIDELINES:
- Use the medical knowledge base above
- Plain language — explain like talking to a friend
- Never diagnose — use "may indicate" or "could be related to"
- Always recommend consulting a healthcare provider
- Format with headers and bullet points"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful health information assistant. You explain lab results using provided medical knowledge. You never diagnose conditions. You always encourage patients to consult their healthcare provider. You use plain, simple language."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return {
            'success': True,
            'analysis_text': response.choices[0].message.content,
            'abnormal_count': len(abnormal_results),
            'abnormal_tests': abnormal_results,
            'knowledge_sources_used': list(retrieved_knowledge.keys()),
            'model': 'GPT-4 with RAG',
            'disclaimer': 'This analysis uses medical knowledge databases combined with AI. It is for informational purposes only and does not constitute medical advice. Always consult with your healthcare provider.'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'analysis_text': f"Found {len(abnormal_results)} abnormal results. Please consult your healthcare provider.",
            'abnormal_count': len(abnormal_results),
            'abnormal_tests': abnormal_results,
            'disclaimer': 'This analysis is for informational purposes only.'
        }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    lab_data = load_sample_labs()
    analysis = analyze_labs_with_gpt(lab_data)
    return render_template('index.html',
                           patient=lab_data['patient'],
                           labs=lab_data['lab_results'],
                           analysis=analysis,
                           test_date=lab_data['test_date'])

@app.route('/api/labs')
def api_labs():
    """Returns raw lab data"""
    return jsonify(load_sample_labs())

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Full RAG analysis for uploaded lab data"""
    lab_data = request.json
    return jsonify(analyze_labs_with_gpt(lab_data))

@app.route('/api/nudges', methods=['GET', 'POST'])
def api_nudges():
    """
    Personalized weekly health nudges endpoint.

    GET:  Uses default sample lab data (demo/testing)
    POST: Accepts custom lab data JSON in request body

    Response:
    {
        "generated_at": "ISO timestamp",
        "patient_id": "MRN",
        "focus_condition": "Iron Deficiency Anemia",
        "nudge_content": {
            "headline": "...",
            "nudges": { "diet": [...], "lifestyle": [...], "talk_to_your_doctor": [...] },
            "encouragement": "..."
        },
        "disclaimer": "..."
    }
    """
    lab_data = request.json if (request.method == 'POST' and request.json) else load_sample_labs()
    try:
        result = generate_health_nudges(lab_data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


# ── FHIR Integration (commented for demo) ─────────────────────────────────────
"""
import requests

def get_patient_labs_via_fhir(patient_id):
    base_url = "https://hapi.fhir.org/baseR4"
    obs_url = f"{base_url}/Observation?patient={patient_id}&category=laboratory&_count=50"
    fhir_data = requests.get(obs_url).json()
    lab_results = []
    if 'entry' in fhir_data:
        for entry in fhir_data['entry']:
            obs = entry['resource']
            lab = {'test_name': obs.get('code', {}).get('text', 'Unknown'),
                   'date': obs.get('effectiveDateTime', 'Unknown')}
            if 'valueQuantity' in obs:
                lab['value'] = obs['valueQuantity']['value']
                lab['unit'] = obs['valueQuantity'].get('unit', '')
                if 'referenceRange' in obs:
                    ref = obs['referenceRange'][0]
                    lab['reference_range'] = f"{ref.get('low',{}).get('value','')} - {ref.get('high',{}).get('value','')}"
            lab_results.append(lab)
    return {'patient_id': patient_id, 'lab_results': lab_results, 'source': 'FHIR'}
"""

if __name__ == '__main__':
    app.run(debug=True)