<p align="center">
  <img src="app/assets/logo.png" alt="VigiSignal-X Logo" width="80"/>
</p>

<h1 align="center">VigiSignal-X</h1>
<p align="center">
  <strong>A Flagship Open-Source Computational Pharmacovigilance Engine & Bipartite Toxicity Network Mapper</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.58+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <em>Detects and prioritizes adverse drug reaction (ADR) safety signal spikes within geriatric polypharmacy cohorts (Age ≥ 60, Concomitant Drug Count > 4) using disproportionality analysis on spontaneous FDA FAERS reporting data.</em>
</p>

---

## 📖 Executive Summary & Strategic Fit

**VigiSignal-X** is an open-source computational pharmacology package designed for large-scale pharmacovigilance signal detection and bipartite risk network mapping. 

The software package provides a scalable pipeline for processing and analyzing spontaneous adverse drug reaction (ADR) reporting data. By integrating high-performance data fetching, advanced statistical disproportionality metrics, bipartite network modeling, and automated clinical report generation, VigiSignal-X enables researchers, clinicians, and regulatory specialists to systematically detect potential safety signals and identify multi-drug interaction patterns without requiring high-performance computing infrastructure or commercial licenses.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A["🌐 openFDA FAERS API"] -->|Async aiohttp + Semaphore| B["core/fda_client.py"]
    B -->|Raw JSON Cache (Large File)| C["data/raw/raw_reports.json"]
    B -->|Pre-Cached Dataset| D["data/processed/processed_reports.csv"]
    
    subgraph Data Triage Configuration
        E1["🟢 Sample Cohort Mode (Offline Cached)"]
        E2["🌐 openFDA API Live Fetch (Rate-Limited)"]
    end
    
    E1 -->|Direct File Load| D
    E2 -->|Live Network Fetch| A

    D --> F["core/stats_engine.py"]
    D --> G["core/network_builder.py"]
    D --> H["core/interaction_matrix.py"]
    D --> I["core/temporal_analysis.py"]
    
    F -->|Triage Priority Classification| J["📊 Tab 1: Safety Signals Dashboard"]
    G -->|NetworkX + Plotly| K["🕸️ Tab 2: Toxicity Network Graph"]
    H -->|Co-Rx Heatmap| L["🔥 Tab 3: Drug Interaction Heatmap"]
    I -->|Quarterly Trends| M["📈 Tab 4: Temporal Signal Trends"]
    F -->|Badged Triage Tiers| N["📋 Tab 5: Analytics Table"]
    F --> O["app/pdf_generator.py"]
    O -->|ReportLab Compiler| P["📄 Tab 6: Clinical PDF Reports"]
    D --> Q["📚 Tab 7: Research Methodology & Insights"]

    R["📄 Clinical Trial Protocol (TXT/PDF)"] --> S["core/protocol_parser.py"]
    S -->|NLP Rule Extraction| T["📋 Tab 8: PharmacoAudit-AI (Clinical Trial Protocol Safety Parser)"]

    U["⚙️ HEOR & Intervention Parameters"] --> V["core/simulation_engine.py"]
    V -->|Markov & Monte Carlo Simulation| W["💸 Tab 9: EconSim-Health (HEOR Polypharmacy Cost-Effectiveness Simulator)"]

    style A fill:#3B82F6,color:#fff,stroke:none
    style B fill:#1E3A8A,color:#fff,stroke:none
    style F fill:#10B981,color:#fff,stroke:none
    style G fill:#F59E0B,color:#fff,stroke:none
    style H fill:#EF4444,color:#fff,stroke:none
    style I fill:#8B5CF6,color:#fff,stroke:none
    style O fill:#6366F1,color:#fff,stroke:none
    style Q fill:#14B8A6,color:#fff,stroke:none
    style S fill:#64748B,color:#fff,stroke:none
    style T fill:#0EA5E9,color:#fff,stroke:none
    style V fill:#059669,color:#fff,stroke:none
    style W fill:#D97757,color:#fff,stroke:none
```

---

## 🧬 Mathematical & Clinical Framework

### 1. Disproportionality Analysis (2×2 Contingency Table)

For each drug-ADR pair, the engine constructs a contingency table comparing reporting frequencies:

| | Target Adverse Event (ADR) | Other Adverse Events |
|---|:---:|:---:|
| **Target Medication** | $a$ (Exposed with Event) | $c$ (Exposed without Event) |
| **Other Medications** | $b$ (Unexposed with Event) | $d$ (Unexposed without Event) |

* **Reporting Odds Ratio (ROR):** Indicates relative reporting rates.
  $$ROR = \frac{a \times d}{b \times c}$$
* **Standard Error (SE) of $\ln(ROR)$:**
  $$SE = \sqrt{\frac{1}{a} + \frac{1}{b} + \frac{1}{c} + \frac{1}{d}}$$
* **95% Confidence Interval (CI):**
  $$95\% \text{ CI} = e^{\ln(ROR) \pm 1.96 \times SE}$$
* **Haldane Correction:** If $a, b, c, \text{ or } d = 0$, the engine automatically adds $0.5$ to all cells to prevent division by zero.

### 2. Clinical Triage Priority Tiers

Instead of binary signal flags, VigiSignal-X uses a custom clinical prioritization matrix:

1. **🔴 High Priority Signal (Tier 1):** $ROR \ge 3.0$ AND $CI_{\text{lower}} > 1.5$ AND cases $a \ge 5$.
2. **🟠 Moderate Priority Signal (Tier 2):** $ROR \ge 2.0$ AND $CI_{\text{lower}} > 1.0$ AND cases $a \ge 3$.
3. **🟡 Weak Priority Signal (Tier 3):** $ROR > 1.0$ AND $CI_{\text{lower}} > 1.0$ AND cases $a \ge 3$.
4. **⚪ Not Significant (Tier 4):** All other drug-ADR combinations.

---

## ✨ Features (9-Tab UI Dashboard)

| Tab | Feature | Description |
|-----|---------|-------------|
| 📊 | **Safety Signals** | Interactive Plotly ROR bar charts grouped by drug and color-coded by Organ System Class. |
| 🕸️ | **Toxicity Network** | Bipartite drug $\rightarrow$ SOC network with layouts (Circular, Bipartite, Spring) color-coded by dominant triage tier. |
| 🔥 | **Drug Interactions** | Co-prescription frequency heatmaps and tabular summaries highlighting polypharmacy hotspots. |
| 📈 | **Temporal Trends** | Aggregated quarterly trend lines and stacked area charts tracking toxicity burdens over time. |
| 📋 | **Analytics Table** | Searchable contingency analytics table equipped with triage tier badges and CSV exports. |
| 📄 | **Clinical Report** | Compiles either a multi-page **Clinical Audit Report** or a compact **One-Page Executive Summary** PDF. |
| 📚 | **Methodology & Insights** | Interactive math formulations (LaTeX), literature-validated positive controls, and limitations. |
| 📋 | **PharmacoAudit-AI (Clinical Trial Protocol Safety Parser)** | NLP-driven clinical trial protocol parsing for programmatically identifying stopping boundaries and safety endpoints. |
| 💸 | **EconSim-Health (HEOR Polypharmacy Cost-Effectiveness Simulator)** | Decision-tree Markov and Monte Carlo simulations measuring health economic cost-effectiveness (ICER, QALYs) of deprescribing. |

---

## ⚡ Setup & Installation

### 1. Clone & Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-username/vigisignal-x.git
cd vigisignal-x

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Test Suite
Ensure that all 105 tests are passing before running the app:
```bash
.venv\Scripts\python -m pytest
```

### 3. Launch the App
```bash
streamlit run app/main.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser. By default, the app starts in **Sample Cohort Mode** using pre-cached datasets so it runs instantly without requiring API keys.

---

## 🖥️ Headless Script Automations

For batch runs or pipeline automation (e.g. running calculations on a remote server), two headless scripts are located under `scripts/`:

1. **Terminal Signal Analyzer:**
   ```bash
   python scripts/run_analysis.py
   ```
   *Loads the cached dataset, filters signals, and outputs the top 15 prioritized signals in a formatted table directly to the console.*

2. **Research Asset Exporter:**
   ```bash
   python scripts/export_research_assets.py
   ```
   *Generates and writes `data/processed/active_signals.csv`, a multi-page `vigisignal_clinical_report_headless.pdf`, and a single-page `vigisignal_executive_summary_headless.pdf` directly to directories without needing the web UI.*

---

## 🧬 Integrated Specialized Modules

VigiSignal-X features two integrated clinical and health economic evaluation modules that work alongside the core safety signal engine:

- **PharmacoAudit-AI (Clinical Trial Protocol Safety Parser)** (Tab 8): An NLP-driven tool designed to parse trial protocols, programmatically extracting inclusion/exclusion criteria, study endpoints, and safety stopping boundaries.
- **EconSim-Health (HEOR Polypharmacy Cost-Effectiveness Simulator)** (Tab 9): A decision-analytic simulation engine that evaluates the quality-adjusted life years (QALYs), health system costs, and Incremental Cost-Effectiveness Ratio (ICER) of pharmacist-led deprescribing interventions.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

<p align="center">
  <strong>Built with ❤️ for medication safety and clinical informatics research</strong>
</p>
