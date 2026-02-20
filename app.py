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

def analyze_labs_with_gpt(lab_data):
    """
    Analyze lab results using GPT-4
    This demonstrates LLM integration with healthcare data
    Production would use Azure OpenAI with BAA for HIPAA compliance
    """
    
    # Build context from lab results
    lab_summary = []
    abnormal_results = []
    
    for lab in lab_data['lab_results']:
        status_marker = f" [{lab['flag']}]" if lab['flag'] else ""
        lab_line = f"- {lab['test_name']}: {lab['value']} {lab['unit']} (Normal: {lab['reference_range']}){status_marker}"
        lab_summary.append(lab_line)
        
        if lab['status'] != 'Normal':
            abnormal_results.append(lab)
    
    # Create prompt with patient context
    prompt = f"""You are a helpful health assistant analyzing lab results. 

Patient Information:
- Name: {lab_data['patient']['name']}
- Test Date: {lab_data['test_date']}

Lab Results:
{chr(10).join(lab_summary)}

Additional Notes: {lab_data.get('notes', 'None')}

Please provide:
1. A brief summary of overall health status (2-3 sentences)
2. Explanation of any abnormal results in plain language
3. 3-5 actionable recommendations
4. What to discuss with their doctor

IMPORTANT GUIDELINES:
- Use simple, non-medical language
- Never diagnose conditions
- Always recommend consulting a healthcare provider
- Be encouraging but honest about concerning results
- Include appropriate medical disclaimers

Format your response clearly with sections."""

    try:
        # Call GPT-4
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful health information assistant. You provide clear explanations of lab results but never diagnose conditions. You always encourage patients to consult their healthcare provider."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        ai_analysis = response.choices[0].message.content
        
        # Structure the response
        return {
            'success': True,
            'analysis_text': ai_analysis,
            'abnormal_count': len(abnormal_results),
            'abnormal_tests': abnormal_results,
            'model': 'GPT-4',
            'disclaimer': 'This analysis is for informational purposes only and does not constitute medical advice. Always consult with your healthcare provider.'
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