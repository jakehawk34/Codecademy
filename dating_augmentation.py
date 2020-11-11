import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
#Create your df here:
df = pd.read_csv('profiles.csv')
#Augment the Data
'''Maps the different types of diets to a scale with religious diets at zero,
vegan at 1, vegetarian at 2, anything at 3, and other at 4'''
diet_mapping = {'halal': 0,
                'strictly halal': 0,
                'mostly halal': 0,
                'kosher': 0,
                'strictly kosher': 0,
                'mostly kosher': 0,
                'vegan': 1,
                'strictly vegan': 1,
                'mostly vegan': 1,
                'vegetarian': 2,
                'strictly vegetarian': 2,
                'mostly vegetarian': 2,
                'anything': 3,
                'strictly anything': 3,
                'mostly anything': 3,
                'other': 4,
                'strictly other': 4,
                'mostly other': 4}
#Maps the different levels of drinking habits
drinks_mapping = {"not at all": 0,
                 "rarely": 1,
                 "socially": 2,
                 "often": 3,
                 "very often": 4,
                 "desperately": 5}
#Maps the different levels of drug use habits
drugs_mapping = {'never': 0,
                 'sometimes': 1,
                 'often': 2}
#Maps the different levels of smoking habits
smokes_mapping = {'no': 0,
                   'sometimes': 1,
                   'when drinking': 1,
                   'trying to quit': 1,
                   'yes': 2}
#Maps the different body types
body_mapping = {"skinny": 0,
                "thin": 1,
                "used up": 2,
                "average": 3,
                "fit": 4,
                "athletic": 5,
                "full figured": 6,
                "jacked": 7,
                "curvy": 8,
                "a little extra": 9,
                "overweight": 10,
                "rather not say": 11}
sex_mapping = {'m': 0, 'f': 1}

education_mapping = {'graduated from college/university': 6,
                     'graduated from masters program': 8,
                     'graduated from two-year college': 4,
                     'graduated from high school': 2,
                     'graduated from ph.d program': 10,
                     'graduated from law school': 10,
                     'college/university': 5,
                     'graduated from med school': 10,
                     'two-year college': 3,
                     'masters program': 7,
                     'high school': 1,
                     'ph.d program': 9,
                     'law school': 9,
                     'med school': 9}


#Make new columns using the mappings above
df['body_code'] = df.body_type.map(body_mapping)
df['diet_code'] = df.diet.map(diet_mapping)
df['drinks_code'] = df.drinks.map(drinks_mapping)
df['drugs_code'] = df.drugs.map(drugs_mapping)
df['smokes_code'] = df.smokes.map(smokes_mapping)
df['sex_code'] = df.sex.map(sex_mapping)
df['educ_code'] = df.education.map(education_mapping)

#List of all the essay columns
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]

#Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
#Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
#Creates a column with the total length of all essays for each profile
df["essay_len"] = all_essays.apply(lambda x: len(x))
#Splits the essays into lists of words
all_essays_split = all_essays.str.split()

#Function to get the average word length
def avg_word_length(essay):
    total_letters = 0
    num_words = 0
    if len(essay) == 0:
        return 0
    else:
        for word in essay:
            total_letters += len(word)
            num_words += 1
        return total_letters / num_words

df["avg_word_length"] = all_essays_split.apply(lambda x: avg_word_length(x))   

#Predict income with sex, age, education, and combined essay length
#Clean the Data
income_data = df[['income', 'sex_code', 'age', 'educ_code', 'essay_len']]
income_features = ['sex_code', 'age', 'educ_code', 'essay_len']
income_data["income_corrected"] = income_data.income.apply(lambda x: df.income.mean() if (x == -1) else x)
income_data_corrected = income_data[["income_corrected", "sex_code", "age", "educ_code", "essay_len"]].reset_index()
filtered_income = income_data_corrected.dropna().reset_index()
income_labels = filtered_income["income_corrected"]
income_predictors = filtered_income[income_features]
print(filtered_income.head())


x_train, x_test, y_train, y_test = train_test_split(income_predictors, income_labels,\
                                                    test_size=0.2, random_state=1)
model = LinearRegression()
model.fit(x_train, y_train)

#MLR of income based on sex, age, education and combined essay length
print('Multiple Linear Regression analysis of income predicted by sex, age, education and total essay length')
print('Train score:', model.score(x_train, y_train))
print("Test score:", model.score(x_test, y_test))
y_predict = model.predict(x_test)

plt.scatter(y_test, y_predict, alpha=0.2)
plt.xlabel('Actual income')
plt.ylabel('Predicted income')
plt.title('Actual Income vs Predicted Income: Multiple Linear Regression with Age, Sex, Education and Combined Length of Essays', fontsize=8)
plt.show()

#KNN Regression of income based on sex, age, education and combined essay length
income_acc = []
k_list = range(1, 31)
for k in k_list:
    regressor_1 = KNeighborsRegressor(n_neighbors = k, weights='distance')
    regressor_1.fit(x_train, y_train)
    income_acc.append(regressor_1.score(x_test, y_test))
    
plt.plot(k_list, income_acc)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('K-Nearest Neighbors Regressor: Income Predcited with Age, Sex, Education, and Combined Length of Essays', fontsize=7)
plt.show()

#Predict body type with age, drinking, smoking, drug use and diet
#Clean the Data
body_data = df[['body_code', 'age', 'diet_code', 'drinks_code', 'smokes_code', 'drugs_code']].reset_index()
body_features = ['age', 'diet_code', 'drinks_code', 'smokes_code', 'drugs_code']
body_data = body_data.dropna().reset_index()
body_labels = body_data["body_code"]
body_predictors = body_data[body_features]
print(body_data.head())

#Normalize the Data
scaler = MinMaxScaler()
scaled_body_type_data = scaler.fit_transform(body_data.values)



x_train, x_test, y_train, y_test = train_test_split\
                                                         (body_predictors,\
                                                          body_labels,\
                                                          test_size=0.2,\
                                                          random_state=1)

#MLR of body type based on age, drinking, smoking, drug use and diet
model = LinearRegression()
model.fit(x_train, y_train)

print('Multiple Linear Regression analysis of body type predicted by age, drinking, smoking, drug use and diet')
print('Train score:', model.score(x_train, y_train))
print("Test score:", model.score(x_test, y_test))
y_predict = model.predict(x_test)

plt.scatter(y_test, y_predict, alpha=0.2)
plt.xlabel('Actual body type')
plt.ylabel('Predicted body type')
plt.title('Actual Body Type vs Predicted Body Type: Multiple Linear Regression with Age, Drinking, Smoking, Drug Use and Diet', fontsize=8)
plt.show()

#KNN Regression of body type based on age, drinking, smoking, drug use and diet
accuracy = []
k_list = range(1, 31)
for k in k_list:
    regressor_2 = KNeighborsRegressor(n_neighbors = k, weights='distance')
    regressor_2.fit(x_train, y_train)
    accuracy.append(regressor_2.score(x_test, y_test))
    
plt.plot(k_list, accuracy)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('K-Nearest Neighbors Regressor: Body Type Predicted with Age, Drinking, Smoking, Drug Use and Diet', fontsize=7)
plt.show()

#KNN Classification of body type based on age, drinking, smoking, drug use and diet
body_acc = []
k_list = range(1, 31)
for k in k_list:
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(x_train, y_train)
    body_acc.append(classifier.score(x_test, y_test))
    
plt.plot(k_list, body_acc)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('K-Nearest Neighbors Classifier: Body Type Predicted with Age, Drinking, Smoking, Drug Use and Diet', fontsize=7)
plt.show()



classifier_2 = KNeighborsClassifier(n_neighbors = 23)
classifier_2.fit(x_train, y_train)
body_max_acc = max(body_acc)
print('Best accuracy from KNN Classifier:')
print(body_max_acc)

print("accuracy score: %s" % accuracy_score(y_test, classifier_2.predict(x_test)))
print("recall score: %s" % recall_score(y_test, classifier_2.predict(x_test), average=None))
print("precision score: %s" % precision_score(y_test, classifier_2.predict(x_test), average=None))
print("f1 score: %s" %f1_score(y_test, classifier_2.predict(x_test), average=None))

# MultinomialNB Classifier for body type
classifiermnb = MultinomialNB()
classifiermnb.fit(x_train, y_train)
print("Multinomial NB classifier for body type based on age and lifestyle:")
print(classifiermnb.score(x_test, y_test))



# can we use an SVC to pick body type?
# loop through different gamma values and relationship of gamma value to score
'''gamma_values = np.arange(0.1, 1.5, 0.1)
scores = []
gamma = 0
score = 0
for g in gamma_values:
    svc_classifier = SVC(kernel='rbf', gamma=g)
    svc_classifier.fit(x_train, y_train)
    current_score = svc_classifier.score(x_test, y_test)
    scores.append(current_score)
    # record highest score and gamma that gives us the highest score
    if current_score > score:
        gamma = g
        score = current_score

plt.plot(gamma_values, scores)
plt.xlabel('Accuracy')
plt.ylabel('Gamma')
plt.title('SVC Classifier: Sex Predicted with Income and Education')
plt.show()

print('Sex SVC Score:')
print('Highest SVC Score: %s' % score)'''

svc_classifier = SVC(kernel='rbf', gamma=1)
svc_classifier.fit(x_train, y_train)

svc_training_score = svc_classifier.score(x_train, y_train)
print("SVC training set score: %s" % svc_training_score)

svc_test_score = svc_classifier.score(x_test, y_test)
print("SVC test set score: %s" % svc_test_score)

svc_predictions = svc_classifier.predict(x_test)
print(confusion_matrix(y_test, svc_predictions))
print(classification_report(y_test, svc_predictions))

#Creates labels and counts for the top five ethnicity profile answers
major_ethnicities_labels = df['ethnicity'].value_counts().index[:5].tolist()
major_ethnicities_counts = df['ethnicity'].value_counts()[:5]
#Creates labels and counts for the top five job profile answers
top_job_labels = df['job'].value_counts().index[:5].tolist()
top_job_counts = df['job'].value_counts()[:5]
#Visualize the Data

#Plots
fig1 = plt.figure(figsize=(8, 8))

ax1 = fig1.add_subplot(2, 2, 1)
plt.scatter(df.age, df.income, alpha=0.2)
plt.title('Income as a function of Age')
ax1.set_xlabel('Age')
ax1.set_ylabel('Income')
ax1.set_xlim(16, 80)
ax1.set_ylim(-1000, 110000)

ax2 = fig1.add_subplot(2, 2, 2)
plt.hist(df.income, bins = 100, color='green')
plt.title('Income Histogram')
ax2.set_yticks([10000, 20000, 30000, 40000])
ax2.set_xlim(-1, 100000)
ax2.set_xlabel('Income')
ax2.set_ylabel('Frequency')

ax3 = fig1.add_subplot(2, 2, 3)
plt.bar(range(5), top_job_counts,\
        color=['green','yellow','orange','red','purple'])
plt.title('Top Jobs')
ax3.set_xticks(range(5))
ax3.set_xticklabels(['Other','Student', 'STEM', 'Computer', 'Fine Arts'])
plt.xticks(rotation=45)
ax3.set_xlabel('Job')
ax3.set_ylabel('Frequency')

ax4 = fig1.add_subplot(2, 2, 4)
plt.pie(major_ethnicities_counts,\
        labels = major_ethnicities_labels,\
        autopct='%d%%')
plt.title('Ethnicity Distribution')
plt.axis('equal')

fig1.tight_layout()

plt.show()
