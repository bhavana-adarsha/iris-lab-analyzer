# IRIS Patient Portal

AI-powered women's health platform prototype using GPT-4 and FHIR integration.

## Overview
Healthcare AI system that retrieves patient data via FHIR standards and provides personalized health guidance through LLM integration.

## Tech Stack
- Backend: Python/Flask
- Healthcare API: FHIR (HL7 standard) (code commented as I decided not to fetch lab results from FHIR server)
- LLM: GPT-4
- Data Source: HAPI FHIR test server

## Key Features
- FHIR-based patient data retrieval (observations, vitals, lab results)
- Patient health record viewing
- AI-powered health guidance system
- Healthcare interoperability using HL7 FHIR standards

## Healthcare Compliance Considerations
Built with HIPAA-compliant deployment in mind:
- FHIR integration patterns for EHR interoperability
- Currently uses synthetic test data from public FHIR server

## Running the Project
# Install dependencies
pip3 install -r requirements.txt

# Run backend
python3 app.py

