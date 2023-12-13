import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 


brownsville_data = pd. read_csv('./brownsville.csv')

X= brownsville_data['sqft', 'bedrooms', 'bathroom', 'year_built']
y= brownsville_data['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = LinearRegression()

y_pred = model. predict (X_test)

mse = mean_squared_error(y_test, y_pred)

r2= r2_score(y_test, y_pred)
print ("Brownsville Regression Results:")
print ("Mean Squared Error:", mse)
print ("R-squared:", r2)