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

---

## Product Roadmap

### Phase 1 — Foundation (Current)
- FHIR-based patient data retrieval using HL7 standards
- AI-powered health guidance using GPT-4
- Patient health record viewing
- Synthetic test data via HAPI FHIR public server
- HIPAA-compliant architecture patterns

---

### Phase 2 — Core Product (3-6 Months)
- Patient authentication and secure login
- Connect to real FHIR servers for live EHR data
- Lab result trend analysis with historical comparisons
- Personalized nutrition and diet suggestions based on lab results
- Improved AI prompt engineering for more accurate health guidance

---

### Phase 3 — Expanded Features (6-12 Months)
- Medication tracking and interaction alerts
- Appointment scheduling integrated with provider calendars
- Push notifications for lab results and health reminders
- Patient health score and wellness dashboard
- Provider dashboard for care team visibility

---

### Phase 4 — Scale & Reach (12-24 Months)
- Mobile app version (iOS and Android)
- Multi-language support for diverse patient populations
- Integration with wearables and remote monitoring devices
- Population health analytics for providers
- Full HIPAA compliance certification and BAA agreements with cloud providers

---

## Contributing

This is a portfolio project. Pull requests are not currently accepted but feel free to fork and build on it.

---

## About

Built by [Bhavana Adarsha](https://github.com/bhavana-adarsha) — Product Leader and AI Builder.

Portfolio: [bhavana-adarsha.github.io](https://bhavana-adarsha.github.io)
LinkedIn: [linkedin.com/in/bhav515](https://www.linkedin.com/in/bhav515/)

