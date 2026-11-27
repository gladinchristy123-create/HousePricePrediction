\# California House Price Prediction Application



\## 1. Project Overview



This project develops a machine learning application to predict estimated median house values using the California Housing dataset.



Three regression algorithms are implemented:

\- Linear Regression

\- Ridge Regression

\- Random Forest Regression



The application provides an interactive Streamlit interface for entering housing details and generating predictions.



\## 2. Technologies Used



\- Python

\- Pandas and NumPy

\- Scikit-learn

\- Matplotlib and Seaborn

\- Streamlit

\- SQLite

\- MLflow

\- Docker

\- Pytest



\## 3. Project Features



\- Exploratory Data Analysis (EDA)

\- Data preprocessing and feature scaling

\- Three regression models

\- Model evaluation using MAE, RMSE and R²

\- Interactive house price prediction

\- Model comparison and visualizations

\- SQLite prediction history

\- MLflow experiment tracking

\- Model versioning

\- Automated prediction tests

\- Docker deployment

\- Basic prediction monitoring



\## 4. Installation



Install Python 3.12 and Docker Desktop.



Create a virtual environment:



```bash

python -m venv venv

```



Activate it in Windows PowerShell:



```powershell

.\\venv\\Scripts\\Activate.ps1

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\## 5. Running the Application



Run:



```bash

streamlit run app.py

```



Open:



http://localhost:8501



Enter housing details, select a regression model and click the prediction button.



\## 6. Model Training



Run:



```bash

python train.py

```



This script:

\- Loads the California Housing dataset.

\- Performs data exploration and preprocessing.

\- Trains Linear Regression, Ridge Regression and Random Forest.

\- Evaluates model performance.

\- Saves fitted preprocessing and model pipelines.

\- Generates visualizations and comparison results.

\- Tracks experiments using MLflow.



\## 7. Model Evaluation



The models are evaluated using:



\- \*\*MAE:\*\* Mean Absolute Error.

\- \*\*RMSE:\*\* Root Mean Squared Error.

\- \*\*R²:\*\* Coefficient of determination.



Results are stored in:



`results/model\_comparison.csv`



\## 8. MLflow Experiment Tracking



The project uses MLflow with a SQLite tracking database.



Start the MLflow interface:



```bash

mlflow ui

```



Open:



http://localhost:5000



MLflow records experiment parameters, evaluation metrics and model artifacts.



\## 9. Database



The application uses SQLite to store:

\- Prediction date and time.

\- Housing input details.

\- Selected model.

\- Predicted house price.

\- Deployed model version.



Prediction history is displayed in the Streamlit application.



\## 10. Model Versioning and Replacement



The deployed model pipeline version is stored in:



`models/model\_metadata.json`



Each prediction records the deployed model version.



\### Procedure to deploy a new model version



1\. Update the training configuration or model parameters in `train.py`.

2\. Run the training script:



```bash

python train.py

```



3\. Evaluate the newly trained models using MAE, RMSE and R².

4\. Select the model suitable for deployment.

5\. Ensure the new fitted preprocessing and model pipeline is saved in the corresponding file under `models/`.

6\. Update the model version in `models/model\_metadata.json`.

7\. Restart or rebuild the application so it loads the updated model files.



For Docker deployment, run:



```bash

docker compose up -d --build

```



8\. Test the new model through the Streamlit application.

9\. Verify that predictions are valid and that the new model version is recorded in the database.



\*\*Important:\*\* Keep a backup of the previous model files and metadata so the previous version can be restored if the new model performs poorly.



\## 11. Representative Housing Profile Testing



Automated tests are implemented in:



`tests/test\_model.py`



The tests evaluate three representative housing profiles:

\- Lower Income Area

\- Average Suburban Area

\- Higher Income Coastal Area



All three regression models are tested against these profiles, giving nine automated prediction tests.



Run:



```bash

python -m pytest -v

```



\## 12. Docker Deployment



Make sure Docker Desktop is running.



Build and start the application:



```bash

docker compose up -d --build

```



Open:



http://localhost:8501



Stop the application:



```bash

docker compose down

```



The Docker Compose configuration mounts the data and models folders to preserve application data and model files outside the container.



\## 13. Monitoring



The application includes a basic monitoring check that detects invalid or negative predictions and displays an alert.



\## 14. Dataset Information



The project uses the California Housing dataset provided by Scikit-learn.



The dataset contains housing and demographic information for California census block groups.



\*\*Note:\*\* Predictions represent estimated median house values for census block groups, not exact market prices of individual properties.



\## 15. Conclusion



This project demonstrates an end-to-end machine learning workflow covering data exploration, preprocessing, regression model development, evaluation, experiment tracking, model versioning, automated testing, database integration, monitoring and Docker deployment.

