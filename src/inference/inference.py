from src.model.predict import predict_sentiment

def get_prediction(text: str):
    """
    Inference helper function for APIs or external calls.
    Returns standard sentiment prediction format.
    """
    try:
        return predict_sentiment(text)
    except Exception as e:
        return {"error": str(e)}
