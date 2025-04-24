## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
### Developed by : Sanjay S
### Reg No : 212223040184
```python
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/431ebc73-1245-497f-8a64-bbe1b78a3cde)
```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![image](https://github.com/user-attachments/assets/dcd693bb-0099-48c7-8f1a-e11fb5f51c9d)

```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![image](https://github.com/user-attachments/assets/6cd44f52-db1a-44d8-8ba9-6d4167141d30)

```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![image](https://github.com/user-attachments/assets/3b6aab3c-24dc-4b5d-bb9b-4703180c7c09)

```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```


```py
df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/user-attachments/assets/50f48da9-be4c-4682-a1e3-005daab469bb)

```py
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/6448c133-49ad-4e91-9831-dca1b63fd98a)

```py
pip install --upgrade category_encoders
```

```py
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```

```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

```py
dfb=pd.concat([df,nd],axis=1)
dfb
```

![image](https://github.com/user-attachments/assets/656e9bed-c4b2-4280-9e02-d86b442ce8b2)

```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![image](https://github.com/user-attachments/assets/dcc2b833-5b36-4ffe-8012-315d2ed17853)

```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```

![image](https://github.com/user-attachments/assets/976d55c5-6c96-44e0-b698-98cb59585c1f)


```py
df.skew()
```

![image](https://github.com/user-attachments/assets/c2f5d571-2879-4c23-b4bf-fa395e27ff29)

```py
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/80c3d75d-e78e-49c3-ac7b-27154c67d577)


```py
np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/user-attachments/assets/70b435f8-b453-46f0-958a-d59f3ba433b5)

```py
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/4af4fdc4-beea-488e-a309-30cdc1c0fbf5)

```py
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/fe11290e-82b7-4b48-9a0c-f96a3bee0415)

```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/user-attachments/assets/b6eb87db-23e0-45eb-b5ff-5481595464bb)

```py
df.skew()
```

![image](https://github.com/user-attachments/assets/f93e8586-15f1-4817-9464-561c52c216f4)

```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

![image](https://github.com/user-attachments/assets/83018ce7-5462-4a81-a0dd-bf69fae1d538)

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

![image](https://github.com/user-attachments/assets/2b814c3c-1e9f-44e3-9886-fca7361013b4)

```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/bdc89f96-7245-4ebb-9ce2-dba7a5fed373)

```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/8b3fe249-b5f9-4736-ae08-727397c5abbd)

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/303d05e5-8782-474a-8d23-cb97b5b8e5c6)

```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/d5fb8240-0086-482f-ba24-cebc8825afa9)

```py
dt=pd.read_csv("titanic_dataset.csv")
dt
```

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```

![image](https://github.com/user-attachments/assets/c53969a5-0a03-4429-a537-c30de337d2f8)

```py
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/250afcf9-9617-4909-b6ea-2dcd38a4173c)


# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
