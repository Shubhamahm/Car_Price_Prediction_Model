import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv("C://Users//shubh//OneDrive//Desktop//All projects//python codes//car_price_dataset.csv")
print(df)

X=df[['Year','Mileage','Engine_Size','Brand_Encoded']]
y=df[['Price']]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("\n Evaluation")
print("mean square error is: ",mean_squared_error(y_test,y_pred))
print("r2 score: ",r2_score(y_test,y_pred))
# print("accuracy score is: ",accuracy_score(y_test,y_pred))
# print("classification report: ",classification_report(y_test,y_pred))

new_car_price=[[2025,100000,3.0,2]]
predicted_price=model.predict(new_car_price)
print("Predicted price of the new car is: ₹", int(predicted_price[0][0]))

# Plotting Feature vs Price (e.g., Year vs Price)

plt.scatter(df['Year'],df['Price'],color='blue',alpha=0.5)
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Car Year vs Price')
plt.grid(True)
plt.show()

# Engine Size vs Price
plt.scatter(df['Engine_Size'], df['Price'], color='red', alpha=0.5)
plt.xlabel('Engine Size (L)')
plt.ylabel('Price')
plt.title('Engine Size vs Price')
plt.grid(True)
plt.show()

# Brand_Encoded vs Price (as Boxplot)

import seaborn as sns

sns.boxplot(x='Brand_Encoded',y='Price',data=df)
plt.xlabel('Brand Encoded')
plt.ylabel('Price')
plt.title('Brand vs Price')
plt.show()

# Actual vs Predicted Prices

plt.scatter(y_test, y_pred, color='purple', alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--')
plt.grid(True)
plt.show()