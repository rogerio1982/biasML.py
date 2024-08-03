import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Criação de um conjunto de dados
np.random.seed(0)
data = {
    'idade': np.random.randint(20, 60, 1000),
    'anos_experiencia': np.random.randint(1, 20, 1000),
    'gênero': np.random.choice(['M', 'F'], 1000),
    'recebeu_promocao': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
}

df = pd.DataFrame(data)

# Conversão de 'gênero' em variável numérica
df['gênero'] = df['gênero'].map({'M': 0, 'F': 1})

# Separação de features e target
X = df.drop(columns=['recebeu_promocao'])
y = df['recebeu_promocao']

# Separação dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 2. Balanceamento das classes no conjunto de treinamento usando SMOTE
smote = SMOTE(random_state=0)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 3. Criação do pipeline com escalonamento e RandomForestClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=0))
])

# Treinamento do modelo
pipeline.fit(X_train_resampled, y_train_resampled)

# 4. Avaliação do modelo
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia geral: {accuracy:.2f}')

# Avaliação por subgrupo (gênero)
for genero in [0, 1]:
    X_test_genero = X_test[X_test['gênero'] == genero]
    y_test_genero = y_test[X_test['gênero'] == genero]
    y_pred_genero = pipeline.predict(X_test_genero)
    acc_genero = accuracy_score(y_test_genero, y_pred_genero)
    tn, fp, fn, tp = confusion_matrix(y_test_genero, y_pred_genero).ravel()
    print(f'Gênero {"Masculino" if genero == 0 else "Feminino"}:')
    print(f'  Acurácia: {acc_genero:.2f}')
    print(f'  False Positive Rate: {fp / (fp + tn):.2f}')
    print(f'  False Negative Rate: {fn / (fn + tp):.2f}')

# 5. Ajuste de limiar de decisão (opcional)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
threshold = 0.5  # Ajuste o limiar conforme necessário
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

# Reavaliação com o novo resultado
accuracy_adjusted = accuracy_score(y_test, y_pred_adjusted)
print(f'Acurácia ajustada: {accuracy_adjusted:.2f}')
