from data_transformation import data_transformation_config
import pandas as pd
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping # type:ignore
import numpy as np
import warnings
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import bentoml

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

import dagshub
dagshub.init(repo_owner='Md-hasan-1', repo_name='Ann-Regression', mlflow=True)

class model_trainer_config:
    def initiate_training(self) -> None:
        warnings.filterwarnings("ignore")
        np.random.seed(40)
    
        # getting train and test data path
        train_data_path = data_transformation_config.train_data_path
        test_data_path = data_transformation_config.test_data_path

        # collection of train and test data 
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)

        # extraction of train test input output features
        X_train, y_train, X_test, y_test = (
            train_data.drop("11", axis=1), train_data["11"], 
            test_data.drop("11", axis=1), test_data["11"]
        )

        X_train = np.array(X_train).astype(np.float32)
        X_test = np.array(X_test).astype(np.float32)
        y_train = np.array(y_train).astype(np.float32)
        y_test = np.array(y_test).astype(np.float32)

        
        # function that creates model 
        def create_model(neurons, layers):
            model = tf.keras.Sequential() # creating a sequential model
            # adding first layer
            model.add(tf.keras.layers.Dense(neurons, 'relu', input_shape=(X_train.shape[1],)))

            # adding other required hidden layers
            for _ in range(layers-1):
                model.add(tf.keras.layers.Dense(neurons, "relu"))
            
            # adding output layer
            model.add(tf.keras.layers.Dense(1))
            
            # adding optimizer, loss function and metrics
            model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['mae'])

            return model

        # parameters of ann regressor 
        params = {
            "neurons" : [16, 32, 64, 120],
            "layers" : [1, 2, 5],
            "epochs" : [50, 100]
        }

        with mlflow.start_run():
            trail_model = KerasRegressor(model=create_model, neurons=32, layers=1,epochs=50, verbose=1)

            grid = GridSearchCV(trail_model, params, n_jobs=-1, cv=3,verbose=1)

            grid.fit(X_train, y_train)

            # creation of final model 
            model = KerasRegressor(neurons=grid.best_params_["neurons"], 
                                    layers=grid.best_params_["layers"], 
                                    build_fn=create_model,
                                    epochs=grid.best_params_["epochs"], 
                                    verbose=1)
            
            patience = (grid.best_params_["epochs"]*30)/100
            # Set up Early Stopping
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

            # training of final model 
            model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=grid.best_params_["epochs"],
                    callbacks=[early_stopping_callback]
                )
            
            # tracking params
            mlflow_tracking_params_dict = dict(
                neurons=grid.best_params_["neurons"], 
                layers=grid.best_params_["layers"], 
                epochs=grid.best_params_["epochs"]
            )
            mlflow.log_params(mlflow_tracking_params_dict)
            mlflow.log_metric("mae", grid.best_score_)
            
            predictions = model.predict(X_train)
            signature = infer_signature(X_train, predictions)

            # For Remote server only(DAGShub)
            remote_server_uri="https://dagshub.com/Md-hasan-1/Ann-Regression.mlflow"
            mlflow.set_tracking_uri(remote_server_uri)
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(
                   model.model_, "KerasRegressor", registered_model_name="KerasRegressor",
                   signature=signature
                )
            else:
                mlflow.keras.log_model(model.model_, "kerasregressor", signature=signature)

            # saving model into dir
            bentoml.keras.save_model("kerasregressor", model.model_)

if __name__=="__main__":
    obj = model_trainer_config()
    obj.initiate_training()
    