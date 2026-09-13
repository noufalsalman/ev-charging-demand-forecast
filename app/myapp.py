import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

import streamlit as st

st.title("Test: Is Streamlit Working?")
st.write("If you can see this, Streamlit is fine and the issue is in the rest of the code.")

# =========================
# 1. DATA LOADING + CACHING
# =========================
@st.cache_data
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # If engineered features are missing, add them
    if "duration_per_ev" not in df.columns:
        df["duration_per_ev"] = df["charging_duration"] / df["no_of_evs_charging"].replace(0, np.nan)
        df["duration_per_ev"] = df["duration_per_ev"].fillna(df["duration_per_ev"].median())

    if "energy_per_min" not in df.columns:
        df["energy_per_min"] = df["energy_consumed"] / df["charging_duration"].replace(0, np.nan)
        df["energy_per_min"] = df["energy_per_min"].fillna(df["energy_per_min"].median())

    if "usage_rate_roll3" not in df.columns:
        df["usage_rate_roll3"] = df["charging_pile_usage_rate"].rolling(3).mean().bfill()

    return df


# =========================
# 2. PREPROCESSING + SPLIT
# =========================
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data(df, target_col: str = "charging_pile_usage_rate"):
    # 1) Keep only numeric columns (drops timestamp, vehicle_type automatically)
    df_num = df.select_dtypes(include=["number"]).copy()

    # 2) Check target exists in numeric df
    if target_col not in df_num.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in numeric columns. "
            f"Available numeric columns: {df_num.columns.tolist()}"
        )

    # 3) Remove leakage + ID columns if they exist
    leakage_features = ["usage_rate_roll3"]        # because it uses the target
    drop_cols = ["charging_station_id"]            # ID column, not useful for ML

    for col in leakage_features + drop_cols:
        if col in df_num.columns:
            df_num = df_num.drop(columns=[col])

    # 4) Build feature list from NUMERIC columns only
    feature_cols = [c for c in df_num.columns if c != target_col]

    # 5) Split into X, y
    X = df_num[feature_cols]
    y = df_num[target_col]

    # 6) Train–test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Feature columns used for training:", feature_cols)
    print("Dtypes:\n", X.dtypes)

    # 7) Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler



# =========================
# 3. TRAIN MODELS
# =========================
def train_models(X_train_scaled, y_train):
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
        ),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "XGBoost": XGBRegressor(
            xgb = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1,
                reg_alpha=0.5,
                eval_metric="rmse",
                random_state=42
)

        ),
    }

    

    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        fitted_models[name] = model

    return fitted_models


def evaluate_models(models, X_test_scaled, y_test):
    results = []
    preds = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        preds[name] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2})

    results_df = pd.DataFrame(results)
    return results_df, preds


# =========================
# 4. STREAMLIT APP LAYOUT
# =========================
def main():
    st.title("EV Charging Pile Usage – ML Dashboard")

    st.sidebar.header("Configuration")
    data_path = st.sidebar.text_input(
        "Dataset path",
        value="charging_pile_demand_forecasting.csv",
        help="Path to your CSV file",
    )

    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview / EDA", "Model Comparison", "Make a Prediction"]
    )

    # Load data
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        st.error("File not found. Please check the dataset path.")
        return

    # Prepare data
    (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        feature_cols,
        scaler,
    ) = prepare_data(df)

    # ================
    # PAGE: OVERVIEW / EDA
    # ================
    if page == "Overview / EDA":
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.markdown("### Basic Statistics")
        st.write(df.describe())

        st.markdown("### Correlation with Target (charging_pile_usage_rate)")
        corr = df.corr(numeric_only=True)["charging_pile_usage_rate"].sort_values(ascending=False)
        st.write(corr)

        st.markdown("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.markdown("### Distributions of Key Features")
        cols_to_plot = ["charging_duration", "energy_consumed", "no_of_evs_charging",
                        "traffic_flow", "charging_pile_usage_rate"]
        fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(20, 4))
        for i, col in enumerate(cols_to_plot):
            sns.histplot(df[col], kde=True, ax=axes[i])
            axes[i].set_title(col)
        st.pyplot(fig)

    # ================
    # PAGE: MODEL COMPARISON
    # ================
    elif page == "Model Comparison":
        st.subheader("Train & Compare Models")

    if st.button("Train Models"):
        # 1. Train models
        models = train_models(X_train_scaled, y_train)
        results_df, preds = evaluate_models(models, X_test_scaled, y_test)

        # 2. Show comparison table
        st.markdown("### Regression Model Comparison")
        st.dataframe(results_df)

        # 3. Metrics bar plots
        st.markdown("### Metrics Comparison")

        models_list = results_df["Model"].tolist()
        r2_values = results_df["R²"].tolist()
        mae_values = results_df["MAE"].tolist()
        rmse_values = results_df["RMSE"].tolist()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # R²
        axes[0].bar(models_list, r2_values)
        axes[0].set_title("R² Score")
        axes[0].set_ylabel("R²")
        axes[0].tick_params(axis="x", rotation=20)

        # MAE
        axes[1].bar(models_list, mae_values)
        axes[1].set_title("Mean Absolute Error (MAE)")
        axes[1].set_ylabel("MAE")
        axes[1].tick_params(axis="x", rotation=20)

        # RMSE
        axes[2].bar(models_list, rmse_values)
        axes[2].set_title("Root Mean Squared Error (RMSE)")
        axes[2].set_ylabel("RMSE")
        axes[2].tick_params(axis="x", rotation=20)

        st.pyplot(fig)

        # 4. Choose BEST MODEL = XGBOOST explicitly (for residual plot and later use)
        if "XGBoost" in models:
            best_model_name = "XGBoost"
        else:
            # Fallback: pick best by RMSE among non-linear models (RF, DT)
            nonlinear_df = results_df[results_df["Model"] != "Linear Regression"]
            best_model_name = nonlinear_df.loc[nonlinear_df["RMSE"].idxmin(), "Model"]

        best_model = models[best_model_name]
        y_pred_best = preds[best_model_name]
        residuals_best = y_test - y_pred_best

        st.markdown(f"### Residual Plot – Best Model: {best_model_name}")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.scatter(y_pred_best, residuals_best, alpha=0.6)
        ax2.axhline(0, color="red", linestyle="--")
        ax2.set_xlabel("Predicted Usage Rate")
        ax2.set_ylabel("Residuals")
        st.pyplot(fig2)

    # ================
    # PAGE: MAKE A PREDICTION
    # ================
    elif page == "Make a Prediction":
        st.subheader("Predict Charging Pile Usage Rate")

        # Train models (or you could load pre-trained)
        models = train_models(X_train_scaled, y_train)
        model_name = st.selectbox("Choose Model", list(models.keys()))
        model = models[model_name]

        st.markdown("#### Input Features")

        user_input = {}
        for col in feature_cols:
            # Skip engineered features if you want — or allow all
            default_val = float(df[col].median())
            user_input[col] = st.number_input(col, value=default_val)

        # Convert to dataframe and scale
        if st.button("Predict"):
            user_df = pd.DataFrame([user_input])
            user_scaled = scaler.transform(user_df[feature_cols])
            y_pred = model.predict(user_scaled)[0]
            st.success(f"Predicted Charging Pile Usage Rate: {y_pred:.3f}")


if __name__ == "__main__":
    main()
