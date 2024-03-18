# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script
Cloning the repository<br>
git clone git@github.com:Dhikshita-macherla/mle-training.git  
<br><br>
Creating and Activating the environment<br>
conda create --name mle-dev <br>
conda activate mle_dev<br>
<br>
Exporting environment file<br>
conda env export >environment.yml<br>
<br>
Executing the python file<br>
python nostandardcode.py<br>
