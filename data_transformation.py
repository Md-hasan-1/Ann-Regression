import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle
import numpy as np
import os

class data_transformation_config:    
    preprocessor_path = os.path.join("artifact/models", "preprocessor.pkl")
    raw_data_path = os.path.join("artifact/data", "raw.csv")
    train_data_path = os.path.join("artifact/data", "train.csv")
    test_data_path = os.path.join("artifact/data", "test.csv")
    
    os.makedirs(os.path.join(os.path.dirname(raw_data_path)), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(preprocessor_path)), exist_ok=True)
    
    # collection and transformation of data
    def initiate_transformation(self) -> None:
        try:
            df = pd.read_csv("data/data.csv") # data collection
        except ImportError:
            raise ImportError
        
        # saving data
        df.to_csv(self.raw_data_path, index=False)
        
        # start process
        df = df.drop(["RowNumber","CustomerId","Surname"], axis=1)

        target_col = "EstimatedSalary"
        num_feature = [feature for feature in df.columns if df[feature].dtype!="O" and feature != target_col]
        cat_feature = [feature for feature in df.columns if df[feature].dtype=="O" and feature != target_col]

        for feature in df.columns:
            if feature !=target_col:
                if feature in num_feature:
                    df[feature].fillna(df[feature].median())
                else:
                    df[feature].fillna(df[feature].mode().iloc[0])

        # input and output feature extraction
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # train test split 
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # initialization of encoders
        oh_encoder = OneHotEncoder(drop="first")
        scaler = StandardScaler()

        # numerical pipelines
        num_pipline = Pipeline(
            steps=[
                ("StandardScaler", scaler)
            ]
        )

        # categorical pipelines
        cat_pipline = Pipeline(
            steps=[
                ("OneHotEncoder", oh_encoder)
            ]
        )

        # initialization of preprocessor
        preprocessor = ColumnTransformer(
            [
                ("num_pipeline", num_pipline, num_feature),
                ("cat_pipline", cat_pipline, cat_feature)
            ]
        )

        # transformation of train data 
        X_train = preprocessor.fit_transform(X_train)

        # transformation of train data 
        X_test = preprocessor.transform(X_test)

        # saving preprocessor object 
        with open(self.preprocessor_path, "wb") as file_obj:
            pickle.dump(preprocessor, file_obj)

        train_data = np.c_[X_train, y_train]
        test_data = np.c_[X_test, y_test]

        # saving transformed train and test data
        pd.DataFrame(train_data).to_csv(self.train_data_path, index=False)
        pd.DataFrame(test_data).to_csv(self.test_data_path, index=False)

if __name__=="__main__":
    dt = data_transformation_config()
    dt.initiate_transformation()
    