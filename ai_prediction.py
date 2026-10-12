
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def load_data(file_path):

    df = pd.read_csv(file_path)

    return df


def preprocess_data(df):

    X = df.drop(columns=["User_ID", "AI_User"])

    y = df["AI_User"]

    return X, y


def train_model(X, y):

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_columns
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000)
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


def predict_ai_user(model, user_data):

    prediction = model.predict(user_data)

    return int(prediction[0])


if __name__ == "__main__":

    df = load_data("AI_User_Prediction_Dataset_600.csv")

    X, y = preprocess_data(df)

    model, accuracy = train_model(X, y)

    print("Model Training Completed")

    print("Model Accuracy:", round(accuracy * 100, 2), "%")

    sample_user = X.iloc[[0]]

    result = predict_ai_user(model, sample_user)

    print("Predicted AI User:", result)
