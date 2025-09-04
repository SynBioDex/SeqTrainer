import json
import argparse
import pandas as pd
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import os


# add more as needed
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
    ap.add_argument("--save_path")
    ap.add_argument("--log_label", default=True)
    args = ap.parse_args()

    print("Arguments:", args)

    data_path = args.data
    data = pd.read_csv(data_path)
    X = data.drop(["y"], axis=1)
    y = (data["y"])

    model = MODEL_MAP[args.model]
    model = model(random_state=args.random_state) if "random_state" in model().get_params() else model()

    with open(args.grid, "r") as f:
        param_grid = json.load(f)

    print("Param grid:", param_grid)


    cv = KFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=args.scoring,
        cv=cv,
        n_jobs=args.n_jobs,
        refit=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=args.random_state
    )

    new_model = TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=np.expm1
    ) if args.log_label else model

    gs = GridSearchCV(
        estimator=new_model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )

    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    print(f"Best CV {args.scoring}: {gs.best_score_:.6f}")

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    with open(os.path.join(args.save_path, f"prediction_metrics_{args.model}.txt"), "w") as f:
        f.write(f"Mean squared error: {mean_squared_error(y_test, y_pred):.6f}\n")
        f.write(f"R^2 score: {r2_score(y_test, y_pred):.6f}\n")
        f.write(f"Mean absolute error: {mean_absolute_error(y_test, y_pred):.6f}\n")
        f.write(f"Root mean squared error: {np.sqrt(mean_squared_error(y_test, y_pred)):.6f}\n")

    # results = pd.DataFrame(gs.cv_results_)
    # results.to_csv(os.path.join(args.save_path, f"results_{args.model}.csv"))

    with open(os.path.join(args.save_path, f"best_params_{args.model}.json"), "w") as f:
        json.dump(gs.best_params_, f, indent=4)

    pickle.dump(best_model, open(os.path.join(args.save_path, f"best_{args.model}.pkl"), "wb"))


if __name__ == "__main__":
    main()
