# CAR-SAR: AI Binder Analysis System

This project is an AI-powered platform to accelerate the design-test-analyze cycle for CAR-T cell binders.

## Core Features
- **Data-driven SAR Analysis:** Upload binder data (sequence, affinity) and PDB structures for analysis.
- **Interactive 3D Visualization:** Inspect binder structures directly within the app.
- **(Upcoming) Affinity Cliff Detection:** Automatically identify key mutations that cause significant changes in binding affinity.
- **(Upcoming) AI-powered Hypothesis Generation:** Leverage LLMs to propose structural and biological reasons for affinity changes.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`