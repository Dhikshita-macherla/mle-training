# mle-training
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

$git clone git@github.com:Dhikshita-macherla/mle-training.git<br>
$cd mle-training<br>
$conda create -f environment.yml<br>
$conda create --name mle-dev<br>
$conda activate mle_dev<br>
$python nonstandardcode.py<br>

## Package build and installation steps
$python -m pip install --upgrade build<br>
$python -m build<br>
$pip install dist/*.whl<br>

### Test the installation
$pytest -v tests/functional_tests/

### Test the unit test
$pytest -v tests/unit_tests/

### Run the application(scripts)
$python scripts/main.py data data/processed .artifacts/model .artifacts/scores

### Look for log
$python scripts/main.py data data/processed .artifacts/model .artifacts/scores --log-path logs

### Docker
$docker build -t dhikshita/docker:v1 .<br>
$docker -p 5008:5008 run dhikshita/docker:v1<br>
$docker pull dhikshita/docker:v1<br>



