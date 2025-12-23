# Commercial Insurance Risk Assessment Network (CIRAN)

CIRAN Underwriter is a lean but realistic LLM‑assisted underwriting pipeline. It ingests a small business profile, 
uses a Google Generative AI chat model to enrich the profile, computes qualitative and quantitative risk signals, and 
emits a simple coverage recommendation.

## Strategic Vision: Transforming Commercial Underwriting

The Commercial Insurance Risk Assessment Network (**CIRAN**) is a cutting-edge, AI-driven automation engine dedicated 
to fundamentally overhauling the process of commercial policy issuance. It moves beyond slow, static underwriting 
reviews to implement a **dynamic, multi-modal risk analysis** that drastically accelerates decision-making while significantly improving risk selection accuracy. This initiative is key to achieving a powerful competitive advantage in the rapidly evolving commercial insurance market.



## Core Value Proposition and Business Impact

CIRAN is engineered to solve the three greatest pain points in commercial underwriting: speed, accuracy, and 
standardization.

1.  **Accelerated Policy Issuance:** By automating the data gathering, analysis, and coverage design steps, the system aims for **60% faster policy issuance**, drastically reducing the current underwriting cycle from days or weeks down to minutes. This enables instant quote generation, dramatically enhancing the client experience.
2.  **Superior Risk Selection:** The system utilizes **parallel, data-driven analysis** to evaluate complex, multi-peril risks. This precision replaces subjective, manual assessments, leading to more accurate premium pricing, which directly translates to an **improved loss ratio** for the carrier.
3.  **Scalable and Consistent Operations:** By codifying the underwriting logic into the **LangGraph** workflow, CIRAN ensures every application is processed against the exact same, up-to-date risk model, guaranteeing consistency and scalability across all underwriting teams and lines of business.


## The Intelligent Agent Workflow: LangGraph Orchestration

The system operates as a **LangGraph State Machine**, where the workflow is defined by the precise handoff of 
data (managed by strict **Pydantic** schemas) between highly specialized agents (nodes). This structure handles 
complex **parallel processing** and conditional logic natively.

### 1. Business Profiler (Data Ingestion & Structuring)
This agent acts as the gateway and data fiduciary.
* **Action:** It receives the initial underwriting request via the **FastAPI** endpoint and immediately simulates calling external data sources using the `requests` library.
* **Output:** The agent cleans, validates, and normalizes disparate data—including financials, operational details, NAICS/SIC industry codes, and location coordinates—into a unified, standardized **State** object, ensuring subsequent agents have clean inputs.

### 2. Parallel Risk Analysis (Hazard & Loss)
The workflow splits at this stage, enabling massive efficiency gains by running two critical, computationally independent tasks simultaneously.

* **Hazard Identifier:**
    * **Function:** Focuses on the **qualitative assessment** of risk factors. It uses simple **Scikit-learn** models (e.g., Logistic Regression) to assess and score specific exposures, such as **Property Hazard Score** (based on construction, location, use) and **Liability Exposure Score** (based on operational scale and industry).
* **Loss Predictor:**
    * **Function:** Focuses on the **quantitative financial outcome**. It employs simple regression models (Scikit-learn Linear Regression) and industry benchmarks to estimate the **Expected Loss (EL)** and **Probable Maximum Loss (PML)** across key coverage lines. This translates the abstract risk scores into tangible financial exposure.

### 3. Coverage Designer (Decision & Recommendation)
This agent serves as the final decision engine, waiting for the parallel streams to **Join** and reconcile their data.

* **Function:** It consumes the consolidated **Hazard Scores** and **Expected Loss Estimates** to formulate the optimal policy structure.
* **Output:** Recommends specific **Coverage Types** (e.g., General Liability, Property, Cyber), determines appropriate **Policy Limits**, suggests necessary **Deductibles**, and outputs the final underwriting recommendation and pricing inputs.

## Project structure

```text
ciran-underwriter/
├─ main.py                     # FastAPI app; wires ChatGoogleGenerativeAI + workflow
├─ config.py                   # Pydantic settings (env‑driven)
├─ schemas.py                  # Request/response models
├─ graph/
│  ├─ workflow.py              # Pydantic WorkflowState + LangGraph builder
│  └─ nodes/
│     ├─ business_profiler.py  # LLM enrichment (NAICS guess, ops summary, risk tags)
│     ├─ hazard_identifier.py  # hazard scores (toy model + short rationale)
│     ├─ loss_predictor.py     # expected loss / PML (toy regression or heuristics)
│     └─ coverage_designer.py  # coverages, limits, deductibles + LLM rationale
├─ services/
│  └─ data_sources.py          # deterministic “external” lookups
├─ utils/
│  └─ logging.py               # simple JSON logging
├─ data/
│  └─ samples/request_example.json
├─ requirements.txt
├─ .env.example
└─ README.md
```

## Configuration

Copy `.env.example` to `.env` and set the following:

| Variable                | Example                   | Notes                                   |
|-------------------------|---------------------------|-----------------------------------------|
| `PORT`                  | `8000`                    | Server port                             |
| `LOG_LEVEL`             | `INFO`                    | Logging level                           |
| `GOOGLE_API_KEY`        | `your_api_key`            | **Required** for ChatGoogleGenerativeAI |
| `GOOGLE_MODEL_NAME`     | `gemini-1.5-flash`        | Model name for the chat LLM             |
| `LLM_TEMPERATURE`       | `0.2`                     | Generation parameter                    |
| `LLM_MAX_OUTPUT_TOKENS` | `2048`                    | Generation parameter                    |

Ensure `config.py` defines these fields (especially `GOOGLE_API_KEY` and `GOOGLE_MODEL_NAME`).

## Running the API

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with your GOOGLE_API_KEY and model

uvicorn main:app --reload
# or: python main.py
```

## Endpoint

**POST `/underwrite`**

Request body (see `schemas.py`; abridged example):

```json
{
  "business_name": "Downtown Bistro",
  "naics_code": null,
  "annual_revenue": 2500000,
  "employee_count": 42,
  "locations": [
    {"line1": "123 Market St", "city": "Austin", "state": "TX", "country": "US"}
  ],
  "property": {"construction": "masonry", "year_built": 1998, "sqft": 15000, "sprinklers": false},
  "operations": {"description": "Full-service restaurant with an open kitchen.", "hours_per_week": 84}
}
```

Example:

```bash
curl -sS -X POST http://localhost:8000/underwrite       -H "Content-Type: application/json"       -d @data/samples/request_example.json | jq .
```

Response shape (abridged):

```json
{
  "hazard_scores": { "property_hazard": 0.68, "liability_exposure": 0.61 },
  "loss_estimates": { "expected_loss": 23500.0, "pml": 117500.0 },
  "recommendation": {
    "coverages": ["Property", "General Liability", "Cyber"],
    "policy_limits": { "Property": 150000.0, "General Liability": 1000000.0, "Cyber": 250000.0 },
    "deductibles": { "Property": 1500.0, "General Liability": 1000.0, "Cyber": 2500.0 },
    "pricing_inputs": { "hazard_factor": 0.65, "loss_load": 23500.0, "pml_load": 117500.0 },
    "rationale": "short LLM-written rationale"
  }
}
```

## How it works

- **Business Profiler** (`graph/nodes/business_profiler.py`) uses the LLM to infer NAICS (if missing), summarize operations, and emit risk tags. Output is merged into `state.profile`.
- **Hazard Identifier** (`graph/nodes/hazard_identifier.py`) produces `property_hazard` and `liability_exposure` in [0, 1] using a tiny model (if present) or heuristics, plus a one‑line rationale.
- **Loss Predictor** (`graph/nodes/loss_predictor.py`) computes `expected_loss` and derives `pml` via a tiny regression (if present) or heuristics.
- **Coverage Designer** (`graph/nodes/coverage_designer.py`) selects coverages and sets limits/deductibles, and asks the LLM for a brief customer‑friendly rationale.
- **Workflow & State** (`graph/workflow.py`) uses a Pydantic `WorkflowState` so LangGraph sees a supported state type and IDEs get stronger typing.


## Troubleshooting

- Ensure `GOOGLE_API_KEY` and `GOOGLE_MODEL_NAME` are set in `.env`.
- If the LLM returns non‑JSON in `business_profiler`, the node falls back to a safe parser.
- If local model files are missing, nodes use deterministic heuristics.
- If your IDE still flags `.state`, verify that your file matches `main.py` in this repository, especially 
- the **lifespan** section.

## License
MIT
