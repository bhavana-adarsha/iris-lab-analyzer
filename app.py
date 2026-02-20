import json
import os
from flask import Flask, jsonify, render_template, request
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def load_sample_labs():
    """Load sample lab results from JSON"""
    with open('sample_data/lab_results.json', 'r') as f:
        return json.load(f)

def load_medical_knowledge():
    """Load medical knowledge base"""
    with open('medical_knowledge.json', 'r') as f:
        return json.load(f)

def retrieve_relevant_knowledge(lab_results, knowledge_base):
    """
    RAG STEP 1: RETRIEVAL
    Retrieve relevant medical knowledge for the specific tests in patient's labs
    This is the 'R' in RAG - we're retrieving context before generation
    """
    relevant_knowledge = {}
    
    for lab in lab_results:
        test_name = lab['test_name']
        
        # Retrieve knowledge for this specific test
        if test_name in knowledge_base['tests']:
            test_info = knowledge_base['tests'][test_name]
            
            # Only include relevant info based on abnormal status
            relevant_info = {
                'description': test_info['description']
            }
            
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

def analyze_labs_with_gpt(lab_data):
    """
    FULL RAG PIPELINE:
    1. RETRIEVE relevant medical knowledge
    2. AUGMENT prompt with retrieved context
    3. GENERATE answer using GPT-4
    """
    
    # Load medical knowledge base
    knowledge_base = load_medical_knowledge()
    
    # RAG STEP 1: RETRIEVE
    retrieved_knowledge = retrieve_relevant_knowledge(lab_data['lab_results'], knowledge_base)
    
    # Build context from lab results
    lab_summary = []
    abnormal_results = []
    
    for lab in lab_data['lab_results']:
        status_marker = f" [{lab['flag']}]" if lab['flag'] else ""
        lab_line = f"- {lab['test_name']}: {lab['value']} {lab['unit']} (Normal: {lab['reference_range']}){status_marker}"
        lab_summary.append(lab_line)
        
        if lab['status'] != 'Normal':
            abnormal_results.append(lab)
    
    # RAG STEP 2: AUGMENT - Build enhanced prompt with retrieved knowledge
    knowledge_context = "\n\nMEDICAL KNOWLEDGE BASE (use this to provide accurate explanations):\n"
    
    for test_name, info in retrieved_knowledge.items():
        knowledge_context += f"\n{test_name}:\n"
        knowledge_context += f"- What it is: {info['description']}\n"
        
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

Please provide a comprehensive analysis using the medical knowledge provided above:

1. **Overall Health Summary** (2-3 sentences)
   
2. **Abnormal Results Explained** 
   For each abnormal result, explain in plain language:
   - What the test measures and why it matters
   - What might cause this result
   - What symptoms to watch for
   
3. **Actionable Recommendations** (5-7 specific next steps)
   - Dietary changes
   - Lifestyle modifications  
   - Follow-up tests to consider
   - When to see a doctor
   
4. **Key Discussion Points for Your Doctor**

IMPORTANT GUIDELINES:
- Use the medical knowledge base provided, don't rely only on your training
- Use simple, non-medical language - explain like talking to a friend
- Never diagnose conditions - say "may indicate" or "could be related to"
- Always recommend consulting a healthcare provider
- Be encouraging but honest about concerning results
- Include appropriate medical disclaimers

Format clearly with headers and bullet points."""

    try:
        # RAG STEP 3: GENERATE using GPT-4 with augmented prompt
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful health information assistant. You explain lab results using provided medical knowledge. You never diagnose conditions. You always encourage patients to consult their healthcare provider. You use plain, simple language."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        ai_analysis = response.choices[0].message.content
        
        # Structure the response
        return {
            'success': True,
            'analysis_text': ai_analysis,
            'abnormal_count': len(abnormal_results),
            'abnormal_tests': abnormal_results,
            'knowledge_sources_used': list(retrieved_knowledge.keys()),
            'model': 'GPT-4 with RAG',
            'disclaimer': 'This analysis uses medical knowledge databases combined with AI. It is for informational purposes only and does not constitute medical advice. Always consult with your healthcare provider.'
        }
        
    except Exception as e:
        # Fallback if API fails
        return {
            'success': False,
            'error': str(e),
            'analysis_text': f"Found {len(abnormal_results)} abnormal results. Please consult your healthcare provider to discuss these findings.",
            'abnormal_count': len(abnormal_results),
            'abnormal_tests': abnormal_results,
            'disclaimer': 'This analysis is for informational purposes only.'
        }

@app.route('/')
def index():
    """Main page - displays lab results and AI analysis"""
    lab_data = load_sample_labs()
    analysis = analyze_labs_with_gpt(lab_data)
    
    return render_template('index.html', 
                         patient=lab_data['patient'],
                         labs=lab_data['lab_results'],
                         analysis=analysis,
                         test_date=lab_data['test_date'])

@app.route('/api/labs')
def api_labs():
    """API endpoint for lab data"""
    return jsonify(load_sample_labs())

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for analysis - allows uploading custom lab data"""
    lab_data = request.json
    analysis = analyze_labs_with_gpt(lab_data)
    return jsonify(analysis)


# ========================================
# FHIR INTEGRATION CODE (COMMENTED FOR DEMO)
# ========================================
# This demonstrates understanding of healthcare interoperability standards
# In production, would fetch real patient data from EHR systems via FHIR

"""
import requests

def get_patient_labs_via_fhir(patient_id):
    '''Fetch patient lab results using HL7 FHIR standard'''
    
    # FHIR server endpoint (HAPI test server)
    base_url = "https://hapi.fhir.org/baseR4"
    
    # Get patient observations (lab results)
    obs_url = f"{base_url}/Observation?patient={patient_id}&category=laboratory&_count=50"
    response = requests.get(obs_url)
    fhir_data = response.json()
    
    # Parse FHIR bundle into simplified format
    lab_results = []
    if 'entry' in fhir_data:
        for entry in fhir_data['entry']:
            obs = entry['resource']
            
            lab = {
                'test_name': obs.get('code', {}).get('text', 'Unknown'),
                'date': obs.get('effectiveDateTime', 'Unknown')
            }
            
            # Extract value and unit
            if 'valueQuantity' in obs:
                lab['value'] = obs['valueQuantity']['value']
                lab['unit'] = obs['valueQuantity'].get('unit', '')
                
                # Determine if abnormal based on reference range
                if 'referenceRange' in obs:
                    ref_range = obs['referenceRange'][0]
                    lab['reference_range'] = f"{ref_range.get('low', {}).get('value', '')} - {ref_range.get('high', {}).get('value', '')}"
            
            lab_results.append(lab)
    
    return {
        'patient_id': patient_id,
        'lab_results': lab_results,
        'source': 'FHIR'
    }

# Example usage:
# patient_data = get_patient_labs_via_fhir('2674930')
"""

if __name__ == '__main__':
    app.run(debug=True)