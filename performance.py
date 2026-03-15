from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

df = load_iris()

X = df.data
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23, stratify=y) #veriyi eğitim ve test olarak böldük. test_size=0.3 ile verinin %30'unu test için ayırdık. random_state=23 ile bölme işleminin her seferinde aynı şekilde yapılmasını sağladık. stratify=y ile sınıf dağılımının eğitim ve test setlerinde aynı olmasını sağladık.
knn = KNeighborsClassifier(n_neighbors=5) #KNN algoritmasını oluşturduk ve k değerini 5 olarak belirledik
knn.fit(X_train, y_train) #modeli eğittik, yani train ettik. X_train ve y_train'i kullanarak modeli oluşturduk.

print(knn.score(X_test, y_test))

