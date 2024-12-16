import bentoml
from bentoml.io import File

KerasRegressorRunner = bentoml.sklearn.get("kerasregressor:latest").to_runner()

svc = bentoml.Service("salary_prediction", runners=[KerasRegressorRunner])

@svc.api(input=File(), output=File())
def streamlit_app(file):
    # This is just a placeholder API. Streamlit serves the app separately.
    return file

