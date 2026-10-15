# Agentic AI for Inventory Optimization

Streamlit app using Walmart M5 data: LightGBM forecasting, Monte Carlo (s, Q) policy simulation, MILP capital allocation, and a Claude assistant for demand shock testing.

MSBA capstone project, College of Business and Economics, California State University, East Bay.
Team: Kaiwalya Mishra, Aditya Chache, Dev Patel.

Store managers have to decide what to reorder, when, and how much, usually with limited working capital and noisy demand. This project chains four models into one Streamlit app to support those decisions on the [M5 Walmart dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy): a LightGBM demand forecast, a Monte Carlo simulation of (s, Q) reorder policies, a MILP that picks one policy per SKU under a capital budget, and a chat assistant that explains the results and runs demand shock tests on request.

The full write-up is in [`docs/Capstone_Final_Report.pdf`](docs/Capstone_Final_Report.pdf) and the slides are in [`docs/Capstone_Final_Presentation.pdf`](docs/Capstone_Final_Presentation.pdf).

## How it works

The app covers 200 SKUs (listed in `selected_products.csv`) across three stores, `CA_1`, `TX_1`, and `WI_1`, which gives 600 SKU-store series.

1. **Forecast** (`backend/forecast_agent.py`). A global LightGBM model trained on lag, rolling, calendar, and price features. In the report it reached a holdout RMSE of 1.7933 and R² of 0.6502, and beat XGBoost, Random Forest, and Linear Regression on validation.
2. **Simulate** (`backend/simulation_agent.py`). Runs 500 Monte Carlo demand paths around the forecast (default demand CV 0.8, 10 day lead time) and reports fill rate, lost sales, average inventory, and order count for a given reorder point `s` and order quantity `Q`.
3. **Optimize one SKU** (`backend/optimization_agent.py`). Searches over (s, Q) and keeps the cheapest policy that meets the target fill rate. Cost combines holding cost (50% of price per year), a fixed $150 per order, and a 1.5x penalty on lost sales.
4. **Optimize the portfolio** (`backend/portfolio_optimizer.py`). Each SKU gets a set of feasible candidate policies, and a Pyomo MILP solved with GLPK selects one per SKU to minimize total cost while keeping average inventory capital under the budget you enter.
5. **Ask the assistant** (`pages/AI_Assistant.py`). Covered below.

The app has three pages: `app.py` (single SKU forecast, simulation preview, and policy results), `pages/Portfolio.py` (budget-constrained portfolio), and `pages/AI_Assistant.py`.

## The AI assistant

This was my main focus on the team.

The assistant does not answer from general knowledge. After you run a single SKU or portfolio optimization, `build_structured_context()` turns those results (policies, simulation metrics, and any shock results) into a structured text block, and that block is sent to Claude (`claude-sonnet-4-6` through the Anthropic SDK) together with the chat history. The system prompt tells the model to use only the numbers in that block, to say so when data is missing, and to write for a non-technical manager in about 4 to 7 sentences.

Messages that look like a demand shock request are handled differently. If you type something like:

```
What happens if demand goes up 30%?
```

the page pulls the percentage out of the message, calls `event_window_shock_tool` (single SKU) or `portfolio_event_window_shock_tool` (portfolio, chosen when the message mentions things like "portfolio" or "budget") from `backend/mcp_tools.py`, re-runs the simulation and optimization with demand scaled up, and shows the baseline and shocked policies side by side with fill rate risk levels. Each shock you run is kept on the page, so you can compare, say, 30% and 50%. If you ask about a shock without giving a number, it asks you for one.

The report found that a +30% shock raised total inventory cost by roughly 38 to 45% for representative SKUs, and +50% raised it by 60 to 80%. In several cases the original policy still met the fill rate target under the shock but needed much more capital, which is the kind of thing the assistant is meant to surface.

One limitation worth knowing: shock detection is keyword based (`is_shock_intent()` matches words like "increase", "demand", or a `%` sign), so an unrelated question containing those words can be routed to the shock tool by mistake.

## Running it locally

> **Needs confirmation.** The app has not been run end to end from this repository, because the data and the trained model are not included (see below). What has been checked: `requirements.txt` installs cleanly into a fresh Python environment, every backend module imports, and the AI Assistant page reads the API key from `.env` and shows an error when it is missing.

```bash
git clone https://github.com/devvdpatel/Inventory-Optimization-using-Agentic-AI.git
cd Inventory-Optimization-using-Agentic-AI

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# The portfolio MILP uses the GLPK solver, which pip does not install.
# On macOS:
brew install glpk

# Add your Anthropic API key
cp .env.example .env    # then edit .env

streamlit run app.py
```

Things you need that are not in the repository:

- **The M5 data.** Download `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv` from the Kaggle competition linked above and place them in `data/raw/m5/`. The `data/` folder is git-ignored.
- **A trained model.** `ForecastAgent` loads `models/model.joblib`, `models/feature_cols.joblib`, and `models/cat_cols.joblib`, and the app will not start without them. You would have to train the model yourself with `notebooks/global_model.ipynb`, but that notebook saves `model_mean.joblib`, `model_q50.joblib`, and `model_q90.joblib` rather than `model.joblib`, so some renaming or code changes are likely needed. *Needs confirmation.*
- **An Anthropic API key.** Claude is the only LLM the app uses, and only on the AI Assistant page. Put the key in `.env` as `ANTHROPIC_API_KEY=...`. The `.env` file is git-ignored; never commit a real key.

The notebooks still contain earlier experiments with Google Gemini, and two unused helpers from that work remain in the backend (`generate_llm_summary()` in `optimization_agent.py` and `parse_intent_with_llm()` in `mcp_tools.py`). The app itself does not call Gemini.

## Repository layout

```
app.py                  Single SKU dashboard (main Streamlit page)
pages/                  Portfolio and AI Assistant pages
backend/                Forecast, simulation, optimization, portfolio MILP, shock tools
notebooks/              Model training and pipeline experiments
selected_products.csv   The 200 SKUs used in the app
proposal/               Original proposal, architecture diagram, policy notes
docs/                   Final report and presentation
```

## Credits

Built by Kaiwalya Mishra, Aditya Chache, and Dev Patel as a team; all three of us worked across the project. The code was originally developed in [adityachache/Agentic-AI-for-Inventory-Optimization](https://github.com/adityachache/Agentic-AI-for-Inventory-Optimization).

Data: M5 Forecasting competition dataset (Walmart), via Kaggle.

No license is attached. The code is shared for viewing as part of a portfolio; please ask before reusing it.
