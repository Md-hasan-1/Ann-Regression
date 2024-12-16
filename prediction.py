import pandas as pd

class Pred:
    def predict(self, model, preprocessor, geography, gender, age, balance, credit_score, 
                Exited, tenure, num_of_products, has_cr_card, 
                is_active_member):
        # Prepare the input data
        input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography':[geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'Exited': [Exited]
        })

        # Scale the input data
        input_data_scaled = preprocessor.transform(input_data)


        # Predict churn
        prediction = model.predict.run(input_data_scaled)
        prediction_proba = prediction[0][0]
                
        return prediction_proba
