from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import precision_score , recall_score,f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')
# print(df.corr(numeric_only=True)
#       ['Outcome'].sort_values(ascending=False))

x = ['Glucose','BMI','Age','Pregnancies']
y = df['Outcome'].values.ravel()

scaler = StandardScaler()
scaler_value = df.copy()
scaler_value[x] = scaler.fit_transform(df[x])

x = scaler_value[x]
y = scaler_value['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)
print(model.coef_)
print("Model prediction result")
y_pre = model.predict(x_test)
print(f'Accuracy Score: {accuracy_score(y_test,y_pre):.4f} * 100')
print(f'Confusion Matrix: ')
print(confusion_matrix(y_test,y_pre))
print(f'Classification Report: ')
print(classification_report(y_test,y_pre))
print('Presision score: ',precision_score(y_test,y_pre))
print('Recall score: ',recall_score(y_test,y_pre))
print('f1 score : ',f1_score(y_test,y_pre))

# cm =confusion_matrix(y_test,y_pre)

# plt.imshow(cm , cmap='Blues')
# plt.colorbar()
# plt.show()

# print("---Your Dibites---")

# Glucose = float(input("Enter your Glucose Level: "))
# BMI = float(input('Enter your BMI: '))
# Age = float(input('Enter Your Age: '))
# Pregnancies = float(input('Enter Your Pregnancies: '))

# data={
#     'Glucose' : Glucose,
#     'BMI' : BMI,
#     'Age' : Age,
#     'Pregnancies': Pregnancies
# }

# data_df = pd.DataFrame([data])

# input_scale = scaler.transform(data_df)
# pred = model.predict(data_df)


# result = 'YOU HAVE DIABETES' if pred == 1 else "CONGRASS YOU DON'T HAVE DIABETES"
# print(f'Your Test Result is : {result} {pred}')