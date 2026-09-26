# **PGR Vesting Decision Support: Project Directives**

## **1\. Architectural Constraints**

* **Language & Style:** Python 3.10+. Enforce strict PEP 8 compliance. Use standard type hinting for all function definitions.  
* **Prohibited Libraries:** Do NOT use `yfinance` for fundamental data or historical ratios. Do NOT use `StandardScaler` across the entire temporal dataset prior to splitting.  
* **Approved Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `xgboost`, `requests`. Utilize Financial Modeling Prep (FMP) or Alpha Vantage REST APIs for robust data ingestion.

## **2\. Quantitative Methodology Rules**

* **Validation Standard:** You MUST NEVER use K-Fold cross-validation. All time-series machine learning models must be validated using strict Walk-Forward Optimization (WFO) utilizing `sklearn.model_selection.TimeSeriesSplit` with defined purge/embargo periods to prevent temporal leakage.  
* **Model Simplicity:** Given the "Large P, Small N" constraint, prioritize high-bias/low-variance algorithms. Default to L1/L2 Regularized Regression (Lasso/Ridge) or extremely shallow tree ensembles to prevent overfitting.  
* **Data Processing:** Compute features on a rolling monthly basis to artificially expand the training dataset size. Utilize unadjusted historical price data to manually calculate total returns, ensuring accurate modeling of stock splits and fractional share DRIP accumulation.  
* **Asset-Specific Logic:** Incorporate PGR's internal "Gainshare" mechanics, utilizing trailing Combined Ratio and PIF growth as leading indicators for the variable annual dividend.

## **3\. Workflow & Verification Requirements**

* **Test-Driven Verification:** Before finalizing any module, you must generate a `pytest` script validating the mathematical outputs. Run the test and verify a passing output before proceeding.

