# Layoff_Predictor_DS
Built a layoff prediction model using machine learning techniques, including feature engineering on company size, market trends, layoffs history, revenue indicators, and workforce attributes. Implemented classification algorithms, performance evaluation, and SHAP-based model explainability to highlight key layoff drivers.

EMPLOYEE LAYOFF PREDICTION MODEL README FILE ~ SUDHARSHAN.S

**MODEL SUMMARY:**

	My project "Employee Layoff Prediction Model" uses a LightGBM Regressor to predict the expected number of employee layoffs based on company, financial, and market features. The model is built using 15 engineered features, including company size, revenue, burn rate, industry growth, and market conditions, with MinMax scaling and Label Encoding applied during preprocessing. A custom Streamlit interface allows users to input company details and get real-time predictions along with risk classifications (Low/Medium/High). The app also provides analytics, feature explanations, and interactive visualizations of layoff trends. This system serves as a data-driven forecasting tool for workforce risk assessment.

**DEPLOYMENT PROCEDURES:**

1 => In the current folder I have uploaded the required files to deploy my model that is 
	i) Layoff Dataset
       ii) My .ipynb File
      iii) My app.py File (Layoff.py)
       iv) After opening the project folder, go to the LAYOFF PREDICTION FINAL.IPYNB and run all the cells in either Jupyter notebook or Google collab and open command prompt and activate the anaconda platform and RUN "streamlit run Layoff.py".
	v) The Above step will deploy the Model in the localhost.

2 => Additionally i have also added a jpg naming "background" which can be fitted as my UI model's background. 
  => You can set the background image by first deploying the model in the localhost  and then click on the ">" on the left side of the website, where you can see the settings of my website in which there will be two options for background that is "Default" and "Custom Image" in which i have fitted the background image.
  => If in case, the current bg image is not to your liking i have also added a folder named "Bg Sample Img" in which i have searched and stored a variety of background images that would suit nicely for my UI.
  => Setting the bg image to your liking is really simple,
	i) Just copy the filename of the bg image you choose and paste it on the "Custom Image's" input tab and also after pasting the filename you should also add ".file extension" of the image you choose.
       ii) In simple if your selected image is a png file the input tab should have "filename.png", for example "layoff_3.png".
      iii) By following the above steps you can change the bg of the UI to your liking.
	

3 => Next is the Prediction Part. 
  => For the User's convenience, i have also added a sample layoff data of 1000 rows from my Primary dataset as "Sample Inputs 1K", which contains 1000 rows of my dataset in the exact same input features as in my Employee Layoff Prediction UI.
  => And in addition, you can also predict future layoffs by giving inputs based on it, like giving "year" input as "2026" and month input as "12" and also other input features based on your wish, and my model will give you the Estimation of predicted Layoffs to the User's Input.
  => Note : The model predictions are not accurate, it can give the average layoff that may happen in the future based on my past layoff dataset.
