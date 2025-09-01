import json
import argparse
import pandas as pd
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

MODEL_MAP = {
    "rfr": RandomForestRegressor,
    "gbr": GradientBoostingRegressor,
    "lr": LinearRegression,
}

def main():
    ap = argparse.ArgumentParser(description="Grid search with log-transformed y.")
    ap.add_argument("--model", required=True, choices=MODEL_MAP.keys())
    ap.add_argument("--grid", required=True,
                    help='Param grid as JSON string or path to JSON file.')
    ap.add_argument("--data", required=True,
                    help="Path to a .csv with X, y (and optional X_test, y_test).")
    ap.add_argument("--cv", type=int, default=5) 
    ap.add_argument("--scoring", default="r2")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--test_split", type=float, default=0.2)
    ap.add_argument("--save_best_to", default=None)
    ap.add_argument("--log_label", default=True)
    args = ap.parse_args()



    data_path = args.data
    data = pd.read_csv(data_path)
    X = data.drop(["y"], axis=1)
    y = data["y"]

    model = MODEL_MAP[args.model]
    model = model(random_state=args.random_state) if "random_state" in model().get_params() else model()

    final_model = TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=np.expm1
    ) if args.log_label else model

    param_grid = json.loads(args.grid)


    cv = KFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    gs = GridSearchCV(
        estimator=final_model,
        param_grid=param_grid,
        scoring=args.scoring,
        cv=cv,
        n_jobs=args.n_jobs,
        refit=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=args.random_state
    )


    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    print(f"Best CV {args.scoring}: {gs.best_score_:.6f}")

    y_pred = gs.predict(X_test)
    print("R² :", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))

if __name__ == "__main__":
    main()
