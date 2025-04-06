# 📊 CPI Analysis & Prediction Dashboard

Welcome to the **CPI (Cost Per Interview) Dashboard**, a Streamlit-powered web application that provides detailed analysis and predictive modeling of survey bid data. It helps users compare won vs. lost deals and recommend optimal CPI pricing strategies.

---

## 🚀 Features

- **Overview Dashboard**: Key metrics and interactive charts summarizing CPI, IR, and LOI differences.
- **CPI Analysis**: Explore CPI variation by Incidence Rate (IR), Length of Interview (LOI), and Sample Size.
- **CPI Prediction**: Machine learning models (Linear Regression, Random Forest, Gradient Boosting) to predict optimal CPI.
- **Actionable Recommendations**: Insights to help pricing teams adjust bids for competitiveness and profitability.

---

## 🗂 Folder Structure

```
Info-Viz/
│
├── app/
│   └── fixed-app-updated.py           # Streamlit main app
│
├── data/
│   ├── invoiced_jobs_*.xlsx           # Won deals data
│   ├── DealItemReportLOST.xlsx        # Lost deals data
│   └── Account+List+with+Segment.csv  # Extra input data
│
├── docs/
│   ├── installation-guide.txt
│   └── readme-file.txt
│
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Ignored files and folders
└── README.md                          # You are here!
```

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/yatharthk2/PredictiveDashboarding.git
cd PredictiveDashboarding

# (Optional) Create virtual environment
python -m venv env
source env/bin/activate  # or .\env\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/fixed-app-updated.py
```

---

## 📁 Data Sources

- `invoiced_jobs_*.xlsx` – Deals that were successfully invoiced
- `DealItemReportLOST.xlsx` – Lost or unconverted bids

---

## 🧠 Models Used

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

All trained on engineered features like `IR/LOI` ratio, `IR/Completes`, and categorical `Type`.

---

## 🙋‍♂️ Author

Developed by **Shriyansh Singh**, **Yatharth Kapadia**, **Mayur Jaisinghani**, **Sakshi Gatyan**, **Tanya Jain**

---

## 📃 License

This project is for educational and demonstration purposes only.
