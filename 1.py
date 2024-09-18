import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data'

column_names = ['Age', 'Sex', 'On_thyroxine', 'Query_on_thyroxine', 'On_antithyroid_medication', 
                'Sick', 'Pregnant', 'Thyroid_surgery', 'I131_treatment', 'Query_hypothyroid', 
                'Query_hyperthyroid', 'Lithium', 'Goitre', 'Tumor', 'Hypopituitary', 'Psych', 
                'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'Class']
data = pd.read_csv(data_url, header=None, names=column_names, na_values=['?'])

data = data.dropna()
data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})
data['Class'] = data['Class'].map({'negative': 0, 'hypothyroid': 1})

X = data.drop(columns=['Class'])
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = BayesianNetwork([('TSH', 'Class'), ('T3', 'Class'), ('TT4', 'Class'), 
                         ('T4U', 'Class'), ('FTI', 'Class')])

model.fit(pd.concat([X_train, y_train], axis=1), estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)

predictions = []
for index, row in X_test.iterrows():
    result = inference.map_query(variables=['Class'], evidence=row.to_dict())
    predictions.append(result['Class'])

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")