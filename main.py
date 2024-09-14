import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Завантажуємо необхідні ресурси NLTK
nltk.download('reuters')
nltk.download('punkt')
nltk.download('stopwords')

# Попередня обробка тексту
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# Доступні категорії в наборі даних Reuters
categories = ['acq', 'crude', 'grain']

# Завантажуємо тексти з обраних категорій
acq_data = [reuters.raw(fileid) for fileid in reuters.fileids(categories='acq')]
crude_data = [reuters.raw(fileid) for fileid in reuters.fileids(categories='crude')]
grain_data = [reuters.raw(fileid) for fileid in reuters.fileids(categories='grain')]

# З'єднуємо всі тексти
texts = acq_data + crude_data + grain_data

# Векторизація текстів
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Визначаємо категорії (0 для 'acq', 1 для 'crude', 2 для 'grain')
y = [0]*len(acq_data) + [1]*len(crude_data) + [2]*len(grain_data)

# Тренувальні і тестові набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель
model = MultinomialNB()
model.fit(X_train, y_train)

# Прогнозування
y_pred = model.predict(X_test)

# Оцінка
print(classification_report(y_test, y_pred, target_names=['acq', 'crude', 'grain']))

# Матриця плутанини
conf_matrix = confusion_matrix(y_test, y_pred)

# Візуалізація результатів
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['acq', 'crude', 'grain'], yticklabels=['acq', 'crude', 'grain'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()