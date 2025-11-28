# 학습 코드 
from sklearn import ensemble

classifier = ensemble.RandomForestClassifier(n_estimators=20, max_depth=3, criterion='gini')