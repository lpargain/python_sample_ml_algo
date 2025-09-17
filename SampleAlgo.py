import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Features - size(sq ft) , bedrooms,age (years)
# Target - price of the house
data = {
    'Size': [1000, 1500, 1200, 2000, 850],
    'Bedrooms': [2, 3, 2, 4, 2],
    'Age': [5, 10, 8, 1, 15],
    'Price': [150000, 200000, 180000, 300000, 120000]  # target

}

# converting data to DataFrame

df = pd.DataFrame(data)

# define features(X) and target (Y)
X = df[['Size', 'Bedrooms', 'Age', 'Price']]
Y = df['Price']

# split the data into training and validation data sets (80% train and 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#  Initialize and train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make prediction on test data

y_predict = model.predict(X_test)

# Evaluate the model

mse = mean_squared_error(y_test, y_predict)  # measure the avg squared difference between actual and predicted values
r2 = r2_score(y_test, y_predict)  # measure how well the regression line approximates

print('Model Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('MSE: %.3f' % mse)
print('R2: %.3f' % r2)

plt.figure(figsize=(10, 6))  # plotting figure -width-10in and height - 6in
#validation
plt.scatter(y_test, y_predict, alpha=0.5)  # actual answer given by ML model
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted Price')
plt.show()
