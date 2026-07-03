# ==========================================
# Loan price  - DATA PREPROCESSING
# ==========================================

import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plot  
import seaborn as sus  
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split  
import joblib

# ===== STEP 1: LOAD DATA =====
raw_data = pd.read_csv(r"C:\Users\Devanshi\Downloads\credit_risk_dataset project1.csv")
data = pd.DataFrame(raw_data)

print("\n" + "="*60)
print("STEP 1: CHECKING FOR MISSING VALUES")
print("="*60)
print("\nMissing value count per column:")
print(data.isnull().sum())

# ===== STEP 2: REMOVE COLUMNS WITH TOO MANY MISSING VALUES =====
data_null_clm = data.isnull().sum() / data.shape[0] * 100  
null20_clm_list = data_null_clm[data_null_clm > 20].index
data_drop_clm = data.drop(columns=null20_clm_list)

print(f"\nRemoved {len(null20_clm_list)} columns with >20% missing values")
print("Remaining missing values:")
print(data_drop_clm.isnull().sum())

# ===== STEP 3: SEPARATE DATA BY TYPE =====
data_object = data_drop_clm.select_dtypes(include=['object'])
data_numaric = data_drop_clm.select_dtypes(include=['int', 'float'])

# ===== STEP 4: HANDLE MISSING VALUES =====
for i in data_numaric.columns:
    data_numaric[i].fillna(data_numaric[i].median(), inplace=True)

for i in data_object.columns:
    data_object[i].fillna(data_object[i].mode()[0], inplace=True)

# ===== STEP 5: ENCODING =====
data_encoded = data_object.copy()

education_mapping = {'High School': 1,'Bachelors': 2,'Masters': 3,'PhD': 4}
data_encoded['Education_Level'] = data_encoded['Education_Level'].map(education_mapping)

data_encoded = pd.get_dummies(data_encoded,columns=['Housing_Status'],drop_first=True,dtype='int')

# ===== SECTION 3: Train_split_test =====
final_data = pd.concat([data_numaric, data_encoded], axis=1)

# 🚨 FIX: Ensure no NaNs remain
final_data = final_data.fillna(0)

X = final_data.drop(columns=['Default'])
y = final_data['Default']

x_train, x_test, y_train_target, y_test_target = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ===== SECTION 4: Scaling =====
scaler = StandardScaler()
scaled_data_final_X_tarin_arr = scaler.fit_transform(x_train)
scaled_data_final_X_tarin = pd.DataFrame(scaled_data_final_X_tarin_arr, columns=x_train.columns)

scaled_data_final_X_test_arr = scaler.transform(x_test)
scaled_data_final_X_test = pd.DataFrame(scaled_data_final_X_test_arr, columns=x_test.columns)

# ===== SECTION 4: Logistic model =====
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(class_weight='balanced', random_state=42)
lr.fit(scaled_data_final_X_tarin,y_train_target)

y_predict = lr.predict(scaled_data_final_X_test)
y_score = lr.score(scaled_data_final_X_test,y_test_target)
print(y_score)

from sklearn.metrics import precision_score,recall_score,f1_score,classification_report,confusion_matrix
print('p_S',precision_score(y_test_target,y_predict))
print('r_s',recall_score(y_test_target,y_predict))
print('r1_socre',f1_score(y_test_target,y_predict))
print("---"*15)
print(confusion_matrix(y_test_target,y_predict))
print("---"*15)
print(classification_report(y_test_target,y_predict))
