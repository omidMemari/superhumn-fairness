import numpy as np
import pandas as pd
import os
from folktables import ACSDataSource, ACSEmployment, ACSIncomePovertyRatio, ACSMobility, ACSIncome, ACSHealthInsurance, ACSPublicCoverage, ACSTravelTime, generate_categories
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from folktables import ACSDataSource, ACSEmployment, ACSIncomePovertyRatio, ACSMobility, ACSIncome, ACSHealthInsurance, ACSPublicCoverage, ACSTravelTime


# import inspect module
import inspect

def create_dataset_ref(dataX, dataA, dataY):
    df = pd.concat([pd.DataFrame(dataX), pd.DataFrame(
        dataA), pd.DataFrame(dataY)], axis=1)
    df.reset_index(inplace=True)
    return df


def main():
    data_source = ACSDataSource(
        survey_year='2018', horizon='1-Year', survey='person')
    west_states = ["CA", "OR", "WA", "NV", "AZ"]
    east_states = ['ME', 'NH', 'MA', 'RI', 'CT', 'NY',
                   'NJ', 'DE', 'MD', 'VA', 'NC', 'SC', 'GA', 'FL']
    acs_data = data_source.get_data(states=west_states, download=True)
    acss = ['acs_west_poverty', 'acs_west_mobility', 'acs_west_income', 'acs_west_insurance',
            'acs_west_public', 'acs_west_travel', 'acs_west_poverty', 'acs_west_employment']
    param = [128, 4, 8, 16, 32, 64, 128, 256]
    folks = [ACSIncomePovertyRatio, ACSMobility, ACSIncome, ACSHealthInsurance,
             ACSPublicCoverage, ACSTravelTime, ACSIncomePovertyRatio, ACSEmployment]
    definition_df = data_source.get_definitions(download=True)
    f = open('to_main.txt', 'w')
    for i in range(len(acss)):
        acs_task, task_name, seed = folks[i], acss[i], param[i]
        #pprint(inspect.getmembers(acs_task))
        group_var = acs_task.group
        target_var = acs_task.target
        groups_to_keep = [1, 2]
        acs_data = acs_data.loc[acs_data[group_var].isin(groups_to_keep)]
        dataX, dataY, dataA = acs_task.df_to_pandas(acs_data)
        # taking catoegorical features        
        categories = generate_categories(features=acs_task.features, definition_df=definition_df)
        categories_cols = categories.keys()
        # taking non-categorical features
        non_categorical_cols = [col for col in dataX.columns if col not in categories_cols]
        dataX[non_categorical_cols] = dataX[non_categorical_cols].astype(float)
        # use standard scaler to scale non_categorical_cols
        pipe = Pipeline([('scaler', StandardScaler())])
        dataX[non_categorical_cols] = pipe.fit_transform(dataX[non_categorical_cols])
        # use one-hot encoding to encode categorical_cols
        dataX = pd.get_dummies(dataX, columns=categories_cols)
        # process dataA and dataY
        dataA = dataA.apply(lambda x: x.astype('category').cat.codes)
        dataY = dataY.apply(lambda x: x.astype('category').cat.codes)
        
            # get name of column A
        f.write("task_name: ")
        f.write('{}'.format(task_name))
        f.write("\n")
        f.write("protected attribute: ")
        f.write('\n')
        f.write('{}'.format(dataA.columns[0]))
        f.write("unique values: ")
        f.write('{}'.format(dataA[dataA.columns[0]].unique()))
        f.write('\n')
        # get name of column Y
        f.write("label: ")
        f.write('{}'.format(dataY.columns[0]))
        f.write("unique values: ")
        f.write('{}'.format(dataY[dataY.columns[0]].unique()))
        f.write('\n\n')
            
        df = create_dataset_ref(dataX, dataA, dataY)
        # # create folder with task_name if not exist
        if not os.path.exists('{}'.format(task_name)):
            os.makedirs('{}'.format(task_name))
        path = '{}/dataset_ref.csv'.format(task_name)
        # df = df.to_numpy()
        # np.savetxt(path, df, delimiter=",")
        df.to_csv(path, index=False)
main()
