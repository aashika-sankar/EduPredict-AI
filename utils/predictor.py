import pandas as pd


def predict_score(model, input_data):
    """
    Predict the student's exam score.

    Parameters:
        model: Trained machine learning model
        input_data: Dictionary containing input values

    Returns:
        Predicted exam score
    """

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    return prediction[0]