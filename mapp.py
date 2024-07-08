import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("Binary Classifier Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your Mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your Mushrooms edible or poisonous? 🍄")

    def load_data():
        data = pd.read_csv("F:/STREAMLIT/mashroom_project/mushrooms.csv")
        LABEL = LabelEncoder()
        for col in data.columns:
            data[col] = LABEL.fit_transform(data[col])
        return data

    def split(df):
        y = df['type']
        X = df.drop(columns=['type'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        return X_train, X_test, y_train, y_test

    def plot_confusion_matrix_custom(model, X_test, y_test, class_names):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix with annotations inside cells
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False,
                         xticklabels=class_names, yticklabels=class_names)

        # Add labels, title, and annotations for TP, FP, FN, TN
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')

        # Loop over data dimensions and create text annotations for each cell
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j + 0.5, i + 0.5, cm[i, j], ha='center', va='center', color='black')

        return plt.gcf()  # Return the current figure explicitly

    def plot_precision_recall_curve_custom(model, X_test, y_test):
        proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        return plt.gcf()  # Return the current figure explicitly

    def plot_roc_curve_custom(model, X_test, y_test):
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        return plt.gcf()  # Return the current figure explicitly

    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    class_names = ["edible", "poisonous"]

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='svm_c')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='svm_kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='svm_gamma')

        metrics = st.sidebar.multiselect("Choose metrics to display:", ("Confusion Matrix", "Precision Recall Curve", "ROC Curve"))

        if st.sidebar.button("Classify", key='svm_classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model.fit(X_train, y_train)

            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)

            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall:", recall_score(y_test, y_pred, labels=class_names).round(2))

            if "Confusion Matrix" in metrics:
                fig_cm = plot_confusion_matrix_custom(model, X_test, y_test, class_names)
                st.pyplot(fig_cm)  # Pass the figure explicitly to st.pyplot()

            if "Precision Recall Curve" in metrics:
                fig_prc = plot_precision_recall_curve_custom(model, X_test, y_test)
                st.pyplot(fig_prc)  # Pass the figure explicitly to st.pyplot()

            if "ROC Curve" in metrics:
                fig_roc = plot_roc_curve_custom(model, X_test, y_test)
                st.pyplot(fig_roc)  # Pass the figure explicitly to st.pyplot()

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='lr_c')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='lr_max_iter')

        metrics = st.sidebar.multiselect("Choose metrics to display:", ("Confusion Matrix", "Precision Recall Curve", "ROC Curve"))

        if st.sidebar.button("Classify", key='lr_classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)

            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)

            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall:", recall_score(y_test, y_pred, labels=class_names).round(2))

            if "Confusion Matrix" in metrics:
                fig_cm = plot_confusion_matrix_custom(model, X_test, y_test, class_names)
                st.pyplot(fig_cm)  # Pass the figure explicitly to st.pyplot()

            if "Precision Recall Curve" in metrics:
                fig_prc = plot_precision_recall_curve_custom(model, X_test, y_test)
                st.pyplot(fig_prc)  # Pass the figure explicitly to st.pyplot()

            if "ROC Curve" in metrics:
                fig_roc = plot_roc_curve_custom(model, X_test, y_test)
                st.pyplot(fig_roc)  # Pass the figure explicitly to st.pyplot()

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step=10, key='rf_n_estimators')
        max_depth = st.sidebar.number_input("Maximum depth of the tree", 1, 20, step=1, key='rf_max_depth')
        bootstrap = st.sidebar.radio("Bootstrap Samples when building trees", (True, False), key='rf_bootstrap')

        metrics = st.sidebar.multiselect("Choose metrics to display:", ("Confusion Matrix", "Precision Recall Curve", "ROC Curve"))

        if st.sidebar.button("Classify", key='rf_classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(X_train, y_train)

            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)

            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall:", recall_score(y_test, y_pred, labels=class_names).round(2))

            if "Confusion Matrix" in metrics:
                fig_cm = plot_confusion_matrix_custom(model, X_test, y_test, class_names)
                st.pyplot(fig_cm)  # Pass the figure explicitly to st.pyplot()

            if "Precision Recall Curve" in metrics:
                fig_prc = plot_precision_recall_curve_custom(model, X_test, y_test)
                st.pyplot(fig_prc)  # Pass the figure explicitly to st.pyplot()

            if "ROC Curve" in metrics:
                fig_roc = plot_roc_curve_custom(model, X_test, y_test)
                st.pyplot(fig_roc)  # Pass the figure explicitly to st.pyplot()

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Dataset (Classification)")
        st.dataframe(df)

if __name__ == '__main__':
    main()
