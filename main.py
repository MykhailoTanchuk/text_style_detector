import nltk
import numpy as np
import os
import re
import string
from collections import Counter
import sys

# Завантаження необхідних ресурсів NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('cmudict')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, cmudict

# Визначення стоп-слів та пунктуації
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Завантаження CMU Pronouncing Dictionary
cmu_dict = cmudict.dict()


# Функція для розбиття тексту на менші частини без перекриття
def segment_text(filename, chunk_size=1000, step=500):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    # Замінюємо всі нові рядки на пробіли
    text = text.replace('\n', ' ')
    # Видаляємо зайві пробіли
    text = ' '.join(text.split())
    # Розбиття на чанки
    chunks = [text[i:i + chunk_size] for i in range(0, len(text) - chunk_size + 1, step)]
    # Переконайтеся, що вибірки унікальні
    unique_chunks = list(set(chunks))
    return unique_chunks

# Функція для завантаження всіх текстів і міток
def load_texts(filenames, styles, chunk_size=1000, step=500):
    texts = []
    labels = []
    for style, filename in zip(styles, filenames):
        if not os.path.exists(filename):
            print(f"Файл {filename} не знайдено. Пропускаємо.")
            continue
        chunks = segment_text(filename, chunk_size, step)
        texts.extend(chunks)
        labels.extend([style] * len(chunks))
    return texts, labels


# Функція для підрахунку складів у слові з використанням CMU Pronouncing Dictionary
def syllable_count(word):
    word = word.lower()
    if word in cmu_dict:
        return [len([y for y in x if y[-1].isdigit()]) for x in cmu_dict[word]][0]
    else:
        # Спрощений підрахунок складів
        vowels = "aeiouy"
        count = 0
        if word and word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count


# Функція для розрахунку індексу Gunning Fog
def gunning_fog_index(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()]
    complex_words = [word for word in words if syllable_count(word) >= 3]
    num_sentences = len(sentences)
    num_words = len(words)
    num_complex_words = len(complex_words)
    if num_sentences == 0 or num_words == 0:
        return 0
    avg_sentence_length = num_words / num_sentences  # Average Sentence Length
    percentage_complex_words = (num_complex_words / num_words) * 100
    fog = 0.4 * (avg_sentence_length + percentage_complex_words)
    return fog


# Функція для витягання додаткових ознак
def get_additional_features(text):
    features = {}

    # Середня довжина слова
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()]
    num_words = len(words)
    avg_word_length = np.mean([len(word) for word in words]) if num_words > 0 else 0
    features['avg_word_length'] = avg_word_length

    # Кількість складних слів
    complex_words = [word for word in words if syllable_count(word) >= 3]
    num_complex_words = len(complex_words)
    features['num_complex_words'] = num_complex_words

    # Розмаїття лексики
    unique_words = len(set(words))
    features['lexical_diversity'] = unique_words / num_words if num_words > 0 else 0

    return features


# Функція для витягання ознак з тексту
def extract_text_features(text):
    features = {}

    # Розбиття тексту на речення
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()]
    num_words = len(words)

    # Середня довжина речення
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
    features['avg_sentence_length'] = avg_sentence_length

    # Кількість питальних речень
    num_questions = len([sent for sent in sentences if sent.strip().endswith('?')])
    features['num_questions'] = num_questions

    # Кількість окличних речень
    num_exclamations = len([sent for sent in sentences if sent.strip().endswith('!')])
    features['num_exclamations'] = num_exclamations

    # Аналіз пунктуації
    punct_counts = Counter(char for char in text if char in punctuation)
    features['num_commas'] = punct_counts.get(',', 0)
    features['num_periods'] = punct_counts.get('.', 0)
    features['num_exclamation_marks'] = punct_counts.get('!', 0)
    features['num_question_marks'] = punct_counts.get('?', 0)
    features['num_colons'] = punct_counts.get(':', 0)
    features['num_semicolons'] = punct_counts.get(';', 0)

    # Індекс читабельності Gunning Fog
    gf_index = gunning_fog_index(text)
    features['gunning_fog'] = gf_index

    # Додаткові ознаки
    additional_feats = get_additional_features(text)
    features.update(additional_feats)

    return features


# Перетворення міток у числовий формат (багатокласова класифікація)
def encode_labels(labels):
    unique_labels = sorted(set(labels))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = np.array([label_mapping[label] for label in labels])
    return encoded, label_mapping


# Нормалізація ознак
def normalize_features(X):
    X = X.astype(float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1  # Уникаємо ділення на нуль
    X_normalized = (X - mean) / std
    return X_normalized


# Реалізація логістичної регресії з градієнтним спуском (One-vs-Rest)
class LogisticRegressionOVR:
    def __init__(self, learning_rate=0.01, epochs=1000, regularization=None, lambda_reg=0.01):
        self.lr = learning_rate
        self.epochs = epochs
        self.regularization = regularization  # 'l1', 'l2' або None
        self.lambda_reg = lambda_reg
        self.classes = None
        self.weights = {}
        self.bias = {}

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit_binary(self, X, y):
        num_samples, num_features = X.shape
        weights = np.zeros(num_features)
        bias = 0

        for epoch in range(self.epochs):
            linear_model = np.dot(X, weights) + bias
            y_predicted = self.sigmoid(linear_model)

            # Обчислення похибки
            error = y_predicted - y
            dw = (1 / num_samples) * np.dot(X.T, error)
            db = (1 / num_samples) * np.sum(error)

            # Додавання регуляризації
            if self.regularization == 'l2':
                dw += (self.lambda_reg / num_samples) * weights
            elif self.regularization == 'l1':
                dw += (self.lambda_reg / num_samples) * np.sign(weights)

            # Оновлення ваг та зсуву
            weights -= self.lr * dw
            bias -= self.lr * db

            # Друк втрат на кожні 100 епох
            if (epoch + 1) % 100 == 0:
                loss = - (1 / num_samples) * np.sum(
                    y * np.log(y_predicted + 1e-15) + (1 - y) * np.log(1 - y_predicted + 1e-15))
                if self.regularization == 'l2':
                    loss += (self.lambda_reg / (2 * num_samples)) * np.sum(weights ** 2)
                elif self.regularization == 'l1':
                    loss += (self.lambda_reg / num_samples) * np.sum(np.abs(weights))
                print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

        return weights, bias

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            print(f"\nТренування для класу: {cls}")
            binary_y = np.where(y == cls, 1, 0)
            weights, bias = self.fit_binary(X, binary_y)
            self.weights[cls] = weights
            self.bias[cls] = bias

    def predict_prob(self, X):
        probs = {}
        for cls in self.classes:
            linear_model = np.dot(X, self.weights[cls]) + self.bias[cls]
            probs[cls] = self.sigmoid(linear_model)
        return probs

    def predict(self, X):
        probs = self.predict_prob(X)
        # Вибираємо клас з найвищою ймовірністю
        predictions = []
        for i in range(X.shape[0]):
            cls_prob = {cls: probs[cls][i] for cls in self.classes}
            predicted_class = max(cls_prob, key=cls_prob.get)
            predictions.append(predicted_class)
        return np.array(predictions)


# Стратифікована k-Fold Крос-валідація з Балансуванням Класів
def manual_stratified_cross_validation(X, y, k=5, random_state=42):
    np.random.seed(random_state)
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Розділення індексів за класами
    classes = np.unique(y)
    class_indices = {cls: indices[y[indices] == cls] for cls in classes}

    # Розподіл індексів класів по фолдам
    class_fold_sizes = {cls: np.full(k, len(class_indices[cls]) // k, dtype=int) for cls in classes}
    for cls in classes:
        class_fold_sizes[cls][:len(class_indices[cls]) % k] += 1

    # Створення фолдів
    folds = []
    current_indices = {cls: 0 for cls in classes}
    for fold in range(k):
        test_indices = []
        for cls in classes:
            start = current_indices[cls]
            end = start + class_fold_sizes[cls][fold]
            test_indices.extend(class_indices[cls][start:end])
            current_indices[cls] = end
        train_indices = np.setdiff1d(indices, test_indices)
        folds.append((train_indices, test_indices))

    metrics = {
        'accuracy': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'precision_micro': [],
        'recall_micro': [],
        'f1_micro': []
    }

    for fold, (train_indices, test_indices) in enumerate(folds):
        print(f"\nФолді {fold + 1}:")
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Балансування класів у тренувальних даних (Random Oversampling)
        X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)

        # Тренування моделі
        model = LogisticRegressionOVR(learning_rate=0.1, epochs=1000, regularization='l2', lambda_reg=0.1)
        model.fit(X_train_balanced, y_train_balanced)

        # Прогнозування
        y_pred = model.predict(X_test)

        # Обчислення метрик
        accuracy = np.mean(y_pred == y_test)
        precision_macro, recall_macro, f1_macro = precision_recall_f1_macro(y_test, y_pred, classes=classes)
        precision_micro, recall_micro, f1_micro = precision_recall_f1_micro(y_test, y_pred)

        print(f"Точність: {accuracy:.4f}")
        print(f"Прецизія (Macro): {precision_macro:.4f}")
        print(f"Відзив (Macro): {recall_macro:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"Прецизія (Micro): {precision_micro:.4f}")
        print(f"Відзив (Micro): {recall_micro:.4f}")
        print(f"F1-Score (Micro): {f1_micro:.4f}")

        metrics['accuracy'].append(accuracy)
        metrics['precision_macro'].append(precision_macro)
        metrics['recall_macro'].append(recall_macro)
        metrics['f1_macro'].append(f1_macro)
        metrics['precision_micro'].append(precision_micro)
        metrics['recall_micro'].append(recall_micro)
        metrics['f1_micro'].append(f1_micro)

    # Обчислення середніх значень та стандартних відхилень метрик
    print("\nСередні результати крос-валідації:")
    for metric in metrics:
        mean = np.mean(metrics[metric])
        std = np.std(metrics[metric])
        print(f"{metric.replace('_', ' ').capitalize()}: {mean:.4f} ± {std:.4f}")


# Функція для балансування класів за допомогою Random Oversampling
def balance_classes(X, y):
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    new_X = []
    new_y = []
    for cls in classes:
        cls_X = X[y == cls]
        cls_y = y[y == cls]
        if len(cls_X) < max_count:
            # Випадкове повторення класу
            additional_indices = np.random.choice(len(cls_X), max_count - len(cls_X), replace=True)
            cls_X_balanced = np.vstack((cls_X, cls_X[additional_indices]))
            cls_y_balanced = np.concatenate((cls_y, cls_y[additional_indices]))
        else:
            cls_X_balanced = cls_X
            cls_y_balanced = cls_y
        new_X.append(cls_X_balanced)
        new_y.append(cls_y_balanced)
    return np.vstack(new_X), np.concatenate(new_y)


# Функції для обчислення метрик вручну (Macro та Micro)
def precision_recall_f1_macro(y_true, y_pred, classes):
    precision = []
    recall = []
    f1 = []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)
    return precision_macro, recall_macro, f1_macro


def precision_recall_f1_micro(y_true, y_pred):
    tp = np.sum(y_true == y_pred)
    fp = np.sum(y_pred == 1) - np.sum(y_true == y_pred)
    fn = np.sum(y_true == 1) - np.sum(y_true == y_pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1_score


# Основна функція
def main():
    styles = ['journalistic', 'literary', 'scientific', 'technical']
    filenames = ['journalistic_generated.txt', 'literary_generated.txt', 'scientific_generated.txt', 'technical_generated.txt']

    texts, labels = load_texts(filenames, styles, chunk_size=1000, step=500)
    print(f"Загальна кількість вибірок: {len(texts)}")

    if not texts:
        print("Немає текстів для аналізу. Перевірте наявність файлів та їхній вміст.")
        return

    # Перетворення міток у числовий формат (багатокласова класифікація)
    encoded_labels, label_mapping = encode_labels(labels)
    print(f"Маппінг міток: {label_mapping}")

    # Витягання ознак для всіх текстів
    feature_list = [extract_text_features(text) for text in texts]

    # Перевірка наявності ознак
    if not feature_list:
        print("Немає ознак для аналізу. Перевірте функцію витягування ознак.")
        return

    # Перетворення ознак у numpy масив
    feature_names = list(feature_list[0].keys())
    X = np.array([[feat[name] for name in feature_names] for feat in feature_list])
    y = encoded_labels

    # Нормалізація ознак
    X = normalize_features(X)

    # Перевірка наявності NaN
    if np.isnan(X).any():
        print("Виявлено NaN значення в ознаках. Заповнення нулями.")
        X = np.nan_to_num(X)

    # Визначення кількості фолдів
    k = 4  # Можна змінити на бажану кількість фолдів
    if len(sys.argv) > 1:
        try:
            k = int(sys.argv[1])
            print(f"Використовується {k}-fold крос-валідація.")
        except ValueError:
            print("Невірний формат аргументу для кількості фолдів. Використовується 5 фолдів.")

    # Стратифікована крос-валідація
    manual_stratified_cross_validation(X, y, k=k, random_state=42)


if __name__ == "__main__":
    main()
