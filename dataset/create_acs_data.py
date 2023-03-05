import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from folktables import ACSDataSource, ACSEmployment, ACSIncomePovertyRatio, ACSMobility, ACSIncome, ACSHealthInsurance, ACSPublicCoverage, ACSTravelTime


# import inspect module
import inspect

def create_dataset_ref(dataX, dataA, dataY):
    df = pd.concat([pd.DataFrame(dataX), pd.DataFrame(
        dataA), pd.DataFrame(dataY)], axis=1)
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

    for i in range(len(acss)):
        acs_task, task_name, seed = folks[i], acss[i], param[i]
        #pprint(inspect.getmembers(acs_task))
        group_var = acs_task.group
        target_var = acs_task.target
        groups_to_keep = [1, 2]
        acs_data = acs_data.loc[acs_data[group_var].isin(groups_to_keep)]
        dataX, dataY, dataA = acs_task.df_to_pandas(acs_data)
        columnsX = dataX.columns
        columnsY = dataY.columns
        columnsA = dataA.columns
        print(dataX.head())
        dataX = preprocessing.normalize(dataX)
        dataA = np.where(dataA == 1, 0, 1)
        dataY = np.where(dataY == 1, 0, 1)
        dataX = pd.DataFrame(dataX, columns=columnsX)
        dataA = pd.DataFrame(dataA, columns=columnsA)
        dataY = pd.DataFrame(dataY, columns=columnsY)
        df = create_dataset_ref(dataX, dataA, dataY)
        # create folder with task_name if not exist
        if not os.path.exists('{}'.format(task_name)):
            os.makedirs('{}'.format(task_name))
        path = '{}/dataset_ref.csv'.format(task_name)
        df.to_csv(path, index=False)
        print(df.head())
main()
