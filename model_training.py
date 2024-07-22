from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_models(X_train, y_train):
    # Initialize the models
    log_reg = LogisticRegression()
    rf_clf = RandomForestClassifier()
    svc_clf = SVC()

    # Fit the models
    log_reg.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train)
    svc_clf.fit(X_train, y_train)

    return log_reg, rf_clf, svc_clf
