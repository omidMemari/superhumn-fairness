import folktables
import numpy as np
import pandas as pd
import os


def create_dataset_ref(dataX, dataA, dataY):
    df = pd.concat([pd.DataFrame(dataX), pd.DataFrame(
        dataA), pd.DataFrame(dataY)], axis=1)
    return df


def main():
    data_source = folktables.ACSDataSource(
        survey_year='2018', horizon='1-Year', survey='person')
    west_states = ["CA", "OR", "WA", "NV", "AZ"]
    east_states = ['ME', 'NH', 'MA', 'RI', 'CT', 'NY',
                   'NJ', 'DE', 'MD', 'VA', 'NC', 'SC', 'GA', 'FL']
    acs_data = data_source.get_data(states=west_states, download=True)
    acss = ['acs_west_poverty', 'acs_west_mobility', 'acs_west_income', 'acs_west_insurance',
            'acs_west_public', 'acs_west_travel', 'acs_west_poverty', 'acs_west_employment']
    param = [128, 4, 8, 16, 32, 64, 128, 256]
    folks = [folktables.ACSIncomePovertyRatio, folktables.ACSMobility, folktables.ACSIncome, folktables.ACSHealthInsurance,
             folktables.ACSPublicCoverage, folktables.ACSTravelTime, folktables.ACSIncomePovertyRatio, folktables.ACSEmployment]

    for i in range(len(acss)):
        acs_task, task_name, seed = folks[i], acss[i], param[i]
        group_var = acs_task.group
        target_var = acs_task.target
        groups_to_keep = [1, 2]
        acs_data = acs_data.loc[acs_data[group_var].isin(groups_to_keep)]
        dataX, dataY, dataA = acs_task.df_to_pandas(acs_data)
        df = create_dataset_ref(dataX, dataY, dataA)
        # create folder with task_name if not exist
        if not os.path.exists('./dataset/{}'.format(task_name)):
            os.makedirs('./dataset/{}'.format(task_name))
        path = './dataset/{}/dataset_ref.csv'.format(task_name)
        df.to_csv(path, index=False)


main()
