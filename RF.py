from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTETomek
import numpy as np
import time
import os


#Fraud.csv
file_time = time.strftime("%Y-%m-%d %H.%M.%S")

#create result folder
result_path = f'Results\\Random Forests\\{file_time}'
if not os.path.exists(result_path):
    os.makedirs(result_path)

# Load the dataset
first = datetime.now()
start = datetime.now()
current_time = start.strftime("%H:%M:%S")
print(f'Loading Dataset [{current_time}]')
df = pd.read_csv("Datasets\\Fraud.csv")

#df = df.drop(columns=['nameOrig', 'step'])


#Edit strings to numbers
df = df.replace(['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER'],[0,1,2,3,4])    #CASH_IN(0), CASH_OUT(1), DEBIT(2), PAYMENT(3) and TRANSFER(4).

# df.loc[df['nameDest'].str.contains('M', case=False), 'nameDest'] = '0'    #Merchant = 0
# df.loc[df['nameDest'].str.contains('C', case=False), 'nameDest'] = '1'    #Customer = 1

df['nameDest'] = df['nameDest'].apply(lambda x: '0' if x[0]=='M' else '1')

df['nameDest'] = df['nameDest'].astype(int)

df['nameOrig'] = df['nameOrig'].apply(lambda x: '0' if x[0]=='M' else '1')

df['nameOrig'] = df['nameOrig'].astype(int)

end = datetime.now()
print(f'Dataset Ready | Time spent: {end-start}')

#Show graph Frauds vs Not Frauds
#print(df['isFraud'].value_counts())
plt.bar([f"Fraud[{df['isFraud'].value_counts()[1]:,}]", f"Not Fraud[{df['isFraud'].value_counts()[0]:,}]"], [df['isFraud'].value_counts()[1], df['isFraud'].value_counts()[0]], color=['blue', 'green'])
plt.xlabel('isFraud')
plt.ylabel('Number of occurences')
plt.title('Number of Frauds/Not Frauds')
plt.show()

frauds_percentage = df['isFraud'].mean()*100
print(f"Fraud Samples Percentage: {frauds_percentage}%")

#Training Dataset
start = datetime.now()
current_time = start.strftime("%H:%M:%S")
print(f'Training Dataset [{current_time}]')

test_size = 0.3
seed = 45

fraud_df = df[df['isFraud'] == 1]
fraud_train, fraud_test = train_test_split(fraud_df, test_size=test_size, random_state=seed)

non_fraud_df = df[df['isFraud'] == 0]
non_fraud_train, non_fraud_test = train_test_split(non_fraud_df, test_size=test_size, random_state=seed)

end = datetime.now()
print(f'Dataset Trained | Time spent: {end-start}')

#Show graph Train/test set
train_df = pd.concat([fraud_train, non_fraud_train])
test_df = pd.concat([fraud_test, non_fraud_test])


train_fraud_count = train_df['isFraud'].sum()
test_fraud_count = test_df['isFraud'].sum()

plt.bar(['Train Set', 'Test Set'], [train_fraud_count, test_fraud_count], color=['blue', 'green'])
plt.xlabel('Dataset')
plt.ylabel('Number of Frauds')
plt.title('Number of Frauds in Train and Test Sets')
plt.show()

X_train = train_df.drop(columns=['isFraud']).to_numpy(dtype=np.float64)
Y_train = train_df['isFraud'].to_numpy(dtype=np.int64)

X_test = test_df.drop(columns=['isFraud']).to_numpy(dtype=np.float64)
Y_test = test_df['isFraud'].to_numpy(dtype=np.int64)

# N_ESTIMATORS = 100      #default 100
# MIN_SAMPLES_SPLIT = 2   #default 2
# MIN_SAMPLES_LEAF = 1    #default 1

for estimators in range(100,300,50):    #trees
    for split in range(2,10):
        for leaf in range(1,5):

            #SMOTE

            #Create Random Forest Model
            start = datetime.now()
            current_time = start.strftime("%H:%M:%S")
            print(f'Creating new Random Forest Classifier [{current_time}]')
            model = RandomForestClassifier(n_estimators=estimators, min_samples_split=split, min_samples_leaf=leaf, class_weight='balanced', n_jobs=-1)
            end = datetime.now()
            print(f'New Random Forest Classifier created | Time spent: {end-start}')

            #Create SMOTE over-sampling method and fit into model
            start = datetime.now()
            current_time = start.strftime("%H:%M:%S")
            print(f'Fit SMOTE oversampling method into model [{current_time}]')
            X_res, y_res = SMOTE().fit_resample(X_train, Y_train)
            print('Resampled dataset shape %s' % Counter(y_res))
            model.fit(X_res, y_res)
            end = datetime.now()
            print(f'SMOTE oversampling method fit into model completed | Time spent: {end-start}')

            #Start predictions
            start = datetime.now()
            current_time = start.strftime("%H:%M:%S")
            print(f'Starting predictions [{current_time}]')
            ypred = model.predict(X_test)
            end = datetime.now()
            print(f'Predictions are done | Time spent: {end-start}')

            print('\nSMOTE:\n')
            report_smote = classification_report(Y_test, ypred, output_dict=True, target_names=['not fraud', 'fraud'])
            print(classification_report(Y_test, ypred, target_names=['not fraud', 'fraud']))
            with open(f'{result_path}\\{file_time}.txt', 'a') as f:
                f.write(f'N_ESTIMATORS: {estimators}\n')
                f.write(f'Split: {split}\n')
                f.write(f'Leaves: {leaf}\n\n')
                f.write('SMOTE:\n')
                f.write(f'\n{report_smote}\n')

            #ADASYN

            #Create Random Forest Model
            start = datetime.now()
            current_time = start.strftime("%H:%M:%S")
            print(f'\nCreating new Random Forest Classifier [{current_time}]')
            model = RandomForestClassifier(n_estimators=estimators, min_samples_split=split, min_samples_leaf=leaf, class_weight='balanced', n_jobs=-1)
            end = datetime.now()
            print(f'New Random Forest Classifier created | Time spent: {end-start}')

            #Create ADASYN over-sampling method and fit into model
            start = datetime.now()
            current_time = start.strftime("%H:%M:%S")
            print(f'Fit ADASYN oversampling method into model [{current_time}]')
            X_res, y_res = ADASYN().fit_resample(X_train, Y_train)
            print('Resampled dataset shape %s' % Counter(y_res))
            model.fit(X_res, y_res)
            end = datetime.now()
            print(f'ADASYN oversampling method fit into model completed | Time spent: {end-start}')

            #Start predictions
            start = datetime.now()
            current_time = start.strftime("%H:%M:%S")
            print(f'Starting predictions [{current_time}]')
            ypred = model.predict(X_test)
            end = datetime.now()
            print(f'Predictions are done | Time spent: {end-start}')

            print('\nADASYN:\n')
            report_adasyn = classification_report(Y_test, ypred, output_dict=True, target_names=['not fraud', 'fraud'])
            print(classification_report(Y_test, ypred, target_names=['not fraud', 'fraud']))
            with open(f'{result_path}\\{file_time}.txt', 'a') as f:
                f.write('ADASYN:\n')
                f.write(f'\n{report_adasyn}\n')


            #EditedNearestNeighbours

            #Create Random Forest Model
            start = datetime.now()
            current_time = start.strftime("%H:%M:%S")
            print(f'\nCreating new Random Forest Classifier [{current_time}]')
            model = RandomForestClassifier(n_estimators=estimators, min_samples_split=split, min_samples_leaf=leaf, class_weight='balanced', n_jobs=-1)
            end = datetime.now()
            print(f'New Random Forest Classifier created | Time spent: {end-start}')

            #Create EditedNearestNeighbours under-sampling method and fit into model
            start = datetime.now()
            current_time = start.strftime("%H:%M:%S")
            print(f'Fit EditedNearestNeighbours undersampling method into model [{current_time}]')
            X_res, y_res = EditedNearestNeighbours().fit_resample(X_train, Y_train)
            print('Resampled dataset shape %s' % Counter(y_res))
            model.fit(X_res, y_res)
            end = datetime.now()
            print(f'EditedNearestNeighbours undersampling method fit into model completed | Time spent: {end-start}')

            #Start predictions
            start = datetime.now()
            current_time = start.strftime("%H:%M:%S")
            print(f'Starting predictions [{current_time}]')
            ypred = model.predict(X_test)
            end = datetime.now()
            print(f'Predictions are done | Time spent: {end-start}')

            print('\nEditedNearestNeighbours:\n')
            report_neighbours = classification_report(Y_test, ypred, output_dict=True, target_names=['not fraud', 'fraud'])
            print(classification_report(Y_test, ypred, target_names=['not fraud', 'fraud']))
            with open(f'{result_path}\\{file_time}.txt', 'a') as f:
                f.write('\nEditedNearestNeighbours:\n')
                f.write(f'\n{report_neighbours}\n')

            #SMOTETomek

            #Create Random Forest Model
            start = datetime.now()
            current_time = start.strftime("%H:%M:%S")
            print(f'\nCreating new Random Forest Classifier [{current_time}]')
            model = RandomForestClassifier(n_estimators=estimators, min_samples_split=split, min_samples_leaf=leaf, class_weight='balanced', n_jobs=-1)
            end = datetime.now()
            print(f'New Random Forest Classifier created | Time spent: {end-start}')

            #Create SMOTETomek combined-sampling method and fit into model
            start = datetime.now()
            current_time = start.strftime("%H:%M:%S")
            print(f'Fit SMOTETomek combined sampling method into model [{current_time}]')
            X_res, y_res = SMOTETomek().fit_resample(X_train, Y_train)
            print('Resampled dataset shape %s' % Counter(y_res))
            model.fit(X_res, y_res)
            end = datetime.now()
            print(f'SMOTETomek combined sampling method fit into model completed | Time spent: {end-start}')

            #Start predictions
            start = datetime.now()
            current_time = start.strftime("%H:%M:%S")
            print(f'Starting predictions [{current_time}]')
            ypred = model.predict(X_test)
            end = datetime.now()
            print(f'Predictions are done | Time spent: {end-start}')

            print('\nSMOTETomek:\n')
            report_smotetomek = classification_report(Y_test, ypred, output_dict=True, target_names=['not fraud', 'fraud'])
            print(classification_report(Y_test, ypred, target_names=['not fraud', 'fraud']))
            with open(f'{result_path}\\{file_time}.txt', 'a') as f:
                f.write('\nSMOTETomek:\n')
                f.write(f'\n{report_smotetomek}\n\n')


            #fraud plot
            labels = ['SMOTE', 'ADASYN', 'Edited\nNearest\nNeighbours', 'SMOTETomek']
            recalls = ["{:.2f}".format(report_smote['fraud']['recall']*100), "{:.2f}".format(report_adasyn['fraud']['recall']*100), "{:.2f}".format(report_neighbours['fraud']['recall']*100), "{:.2f}".format(report_smotetomek['fraud']['recall']*100)]
            precisions = ["{:.2f}".format(report_smote['fraud']['precision']*100), "{:.2f}".format(report_adasyn['fraud']['precision']*100), "{:.2f}".format(report_neighbours['fraud']['precision']*100), "{:.2f}".format(report_smotetomek['fraud']['precision']*100)]
            recalls = [float(x) for x in recalls]
            precisions = [float(x) for x in precisions]


            x = np.arange(len(labels))
            width = 0.35

            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width/2, precisions, width, label='Precision')
            rects2 = ax.bar(x + width/2, recalls, width, label='Recall')

            ax.set_ylabel('Percentage')
            ax.set_title(f'Trees: {estimators}, Min Samples Split: {split}, Min Samples Leaf: {leaf}')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            def autolabel(rects):
                """Attach a text label above each bar in *rects*, displaying its height."""
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate('{}'.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 0),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')


            autolabel(rects1)
            autolabel(rects2)

            plt.tight_layout()

            plt.savefig(f'{result_path}\\fraud_{file_time}({estimators}_{split}_{leaf}).png')

            plt.show()

            #No fraud plot
            labels = ['SMOTE', 'ADASYN', 'Edited\nNearest\nNeighbours', 'SMOTETomek']
            recalls = ["{:.2f}".format(report_smote['not fraud']['recall']*100), "{:.2f}".format(report_adasyn['not fraud']['recall']*100), "{:.2f}".format(report_neighbours['not fraud']['recall']*100), "{:.2f}".format(report_smotetomek['not fraud']['recall']*100)]
            precisions = ["{:.2f}".format(report_smote['not fraud']['precision']*100), "{:.2f}".format(report_adasyn['not fraud']['precision']*100), "{:.2f}".format(report_neighbours['not fraud']['precision']*100), "{:.2f}".format(report_smotetomek['not fraud']['precision']*100)]
            recalls = [float(x) for x in recalls]
            precisions = [float(x) for x in precisions]


            x = np.arange(len(labels))
            width = 0.35

            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width/2, precisions, width, label='Precision')
            rects2 = ax.bar(x + width/2, recalls, width, label='Recall')

            ax.set_ylabel('Percentage')
            ax.set_title(f'Trees: {estimators}, Min Samples Split: {split}, Min Samples Leaf: {leaf}')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            def autolabel(rects):
                """Attach a text label above each bar in *rects*, displaying its height."""
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate('{}'.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 0),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')


            autolabel(rects1)
            autolabel(rects2)

            plt.tight_layout()


            plt.savefig(f'{result_path}\\no_fraud_{file_time}({estimators}_{split}_{leaf}).png')

            plt.show()

            print(f'\nTotal time: {end-first}')
