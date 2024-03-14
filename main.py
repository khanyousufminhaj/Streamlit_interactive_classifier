import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


st.title("Interactive Classifier Streamlit Webpage")

st.write("""# Explore different classifer
         """)
dataset_name=st.sidebar.selectbox(
    "Select Dataset",
    {"Iris","Breast cancer","Wine"})


classifier_name=st.sidebar.selectbox('Select Classifier',
    {"kNN",'SVM',"Random Forest"})

def get_dataset(dataset_name):
    if dataset_name=='Iris':
        data=datasets.load_iris()
    elif dataset_name=='Breast Cancer':
        data=datasets.load_breast_cancer()
    else:
        data=datasets.load_wine()
    X=data.data
    Y=data.target
    return X,Y
X,Y=get_dataset(dataset_name)
st.write("shape of dataset",X.shape)
st.write("number of classes",len(np.unique(Y)))

def add_parameter_ui(clf_name):
    params=dict()
    if clf_name=='kNN':
        k=st.sidebar.slider("k : number of neighbours",1,15)
        params={'k':k}
    elif clf_name=="SVM":
        c=st.sidebar.select_slider("c",list(np.arange(0.01,10.01,0.01)))
        params={'c':c}
    elif clf_name=="Random Forest":
        max_depth=st.sidebar.slider("max depth",2,15)
        n_estimators=st.sidebar.slider("Number of estimators",1,100)
        params={'max_depth':max_depth,"n_estimators":n_estimators}
    return params
params=add_parameter_ui(classifier_name)

def get_classifier(clf_name,params):
    if clf_name=='kNN':
        clf=KNeighborsClassifier(n_neighbors=params['k'])
    elif clf_name=='SVM':
        clf=SVC(C=params['c'])
    elif clf_name=='Random Forest':
        clf=RandomForestClassifier(n_estimators=params['n_estimators'],
                                   max_depth=params['max_depth'])
    return clf
clf=get_classifier(clf_name=classifier_name,params=params)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)
accuracy=accuracy_score(Y_test,Y_pred)
st.write(f"# Classifier={classifier_name}")
st.write(f"# Accuracy score={accuracy}")

pca=PCA(2)
X_projected=pca.fit_transform(X)
x1=X_projected[:,0]
x2=X_projected[:,1]

fig=plt.figure()
plt.scatter(x1,x2,c=Y)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
st.pyplot(fig)#plt.show