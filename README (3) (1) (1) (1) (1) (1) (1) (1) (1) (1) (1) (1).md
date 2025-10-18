# JetLearn MIS – Streamlit App (Pipeline‑aware)

Visualize JetLearn enrolments (payments) as **bubble charts** at two levels:

- **MTD (Same-Month Created & Paid):** `Payment Received Date` is in the selected month **and** `Create Date` is in the same month.
- **Cover (All Payments in Month):** `Payment Received Date` is in the selected month (ignore `Create Date`).

The split beneath each total uses the **`Pipeline`** column (e.g., `AI-Coding Pipeline`, `Maths Pipeline`) to bucket into **AI Coding** vs **Math**.

---

## Files
- `app.py` – Streamlit app
- `requirements.txt` – pinned dependencies
- `Master_sheet_DB.csv` – pre-uploaded dataset (not included here)
- `README.md` – this guide

---

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Data Columns (auto-detected)
- **Create Date** (variants allowed: `Create Date`, `Create_Date`, `Created At`, etc.)
- **Payment Received Date** (variants allowed; trailing spaces are stripped)
- **Pipeline** (used for split; values containing “math” → Math; containing “ai/coding” → AI Coding)

*Note:* The app strips column-name whitespace (e.g., `'Payment Received Date '` → `'Payment Received Date'`). Dates are parsed with day-first preference and UNIX seconds/ms fallback.

---

## What You’ll See
- Sidebar -> **MIS**
- Main panel:
  - **MTD** bubble (Total) + two child bubbles (**AI Coding**, **Math**)
  - **Cover** bubble (Total) + two child bubbles (**AI Coding**, **Math**)
  - KPI strip for quick counts
  - Data preview & detected column mapping