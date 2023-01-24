import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper

#data_path = Path(f"{os.getcwd()}/data")

def main():
    df = (pd.read_csv("diabetic_data.csv")
        .rename(columns={"diag_1": "primary_diagnosis"}))
    # Create Outcome variables
    df.loc[:, "readmit_30_days"] = (df["readmitted"] == "<30")
    df.loc[:, "readmit_binary"] = (df["readmitted"] != "NO")
    # Replace missing values and re-code categories
    df.loc[:,"age"] = df.age.replace({"?": ""})
    df.loc[:,"payer_code"] = df["payer_code"].replace({"?", "Unknown"})
    df.loc[:,"medical_specialty"] = df["medical_specialty"].replace({"?": "Missing"})
    df.loc[:, "race"] = df["race"].replace({"?": "Unknown"})

    df.loc[:, "admission_source_id"] = df["admission_source_id"].replace({1: "Referral", 2: "Referral", 3: "Referral", 7: "Emergency"})
    df.loc[:, "age"] = df["age"].replace( ["[0-10)", "[10-20)", "[20-30)"], "30 years or younger")
    df.loc[:, "age"] = df["age"].replace(["[30-40)", "[40-50)", "[50-60)"], "30-60 years")
    df.loc[:, "age"] = df["age"].replace(["[60-70)", "[70-80)", "[80-90)"], "Over 60 years")
    
    # Clean various medical codes
    df.loc[:, "discharge_disposition_id"] = (df.discharge_disposition_id
                                            .apply(lambda x: "Discharged to Home" if x==1 else "Other"))

    df.loc[:, "admission_source_id"] = df["admission_source_id"].apply(lambda x: x if x in ["Emergency", "Referral"] else "Other")
    # Re-code Medical Specialties and Primary Diagnosis
    specialties = [
        "Missing",
        "InternalMedicine",
        "Emergency/Trauma",
        "Family/GeneralPractice",
        "Cardiology",
        "Surgery"
    ]
    df.loc[:, "medical_specialty"] = df["medical_specialty"].apply(lambda x: x if x in specialties else "Other")
    #
    df.loc[:, "primary_diagnosis"] = df["primary_diagnosis"].replace(
        regex={
            "[7][1-3][0-9]": "Musculoskeltal Issues",
            "250.*": "Diabetes",
            "[4][6-9][0-9]|[5][0-1][0-9]|786": "Respitory Issues",
            "[5][8-9][0-9]|[6][0-2][0-9]|788": "Genitourinary Issues"
        }
    )
    diagnoses = ["Respitory Issues", "Diabetes", "Genitourinary Issues", "Musculoskeltal Issues"]
    df.loc[:, "primary_diagnosis"] = df["primary_diagnosis"].apply(lambda x: x if x in diagnoses else "Other")



    #Binarize and bin features
    df.loc[:, "medicare"] = (df.payer_code == "MC")
    df.loc[:, "medicaid"] = (df.payer_code == "MD")

    df.loc[:, "had_emergency"] = (df["number_emergency"] > 0)
    df.loc[:, "had_inpatient_days"] = (df["number_inpatient"] > 0)
    df.loc[:, "had_outpatient_days"] = (df["number_outpatient"] > 0)

    # Save DataFrame
    cols_to_keep = ["race","gender","age","discharge_disposition_id","admission_source_id","time_in_hospital",
        "medical_specialty","num_lab_procedures","num_procedures","num_medications","primary_diagnosis","number_diagnoses","max_glu_serum","A1Cresult","insulin","change",
        "diabetesMed", "medicare", "medicaid", "had_emergency", "had_inpatient_days", "had_outpatient_days", "readmitted","readmit_binary","readmit_30_days"]

    df = df.loc[:, cols_to_keep]
    
    df.to_csv("diabetic_preprocessed.csv", index=False)
    

    categorical_features = [
        "race",
        "gender",
        "age",
        "discharge_disposition_id",
        "admission_source_id",
        "medical_specialty",
        "primary_diagnosis",
        "max_glu_serum",
        "A1Cresult",
        "insulin",
        "change",
        "diabetesMed",
        "readmitted"]
    
    for col_name in categorical_features:
        df[col_name] = df[col_name].astype("category")
    
    # drop gender group Unknown/Invalid
    #preprocessed_df = df.query("gender != 'Unknown/Invalid'")
    
    df = df[df.gender != 'Unknown/Invalid']
    print(df["gender"].value_counts())
    print(df)
    # retain the original race as race_all, and merge Asian+Hispanic+Other 
    df["race_all"] = df["race"]
    df["race"] = df["race"].replace({"Asian": "Other", "Hispanic": "Other"})
    target_variable = "readmit_30_days"
    demographic = ["race", "gender"]
    sensitive = ["gender"]
    #Y, A = df.loc[:, target_variable], df.loc[:, sensitive]
    Y, A = df[[target_variable]], df[sensitive]

    df = pd.get_dummies(df.drop(columns=[
        "race_all",
        "discharge_disposition_id",
        "readmitted",
        "readmit_binary",
        "readmit_30_days"  # "readmit_30_days" is the label, "race" or "gender" --> protected attribute
    ]))

    # from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    # from sklearn.compose import ColumnTransformer, make_column_transformer
    # preprocess = make_column_transformer(
    #     (['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'], StandardScaler()),
    #     (['gender', 'race', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country', 'over_25','label'], OneHotEncoder(sparse=False))
    # )
    #print(final_df.head())
    # Instantiate encoder/scaler

    #pipe = Pipeline([('scaler', StandardScaler())])
    # The pipeline can be used as any other estimator
    # and avoids leaking the test set into the train set
    #print(df["race"].value_counts())
    #print(df["readmit_30_days"].value_counts())
    scaled_features = StandardScaler().fit_transform(df.values)
    df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    A['gender'].replace({"Female":2, "Male":1}, inplace=True)
    #A['race'].replace({"AfricanAmerican":2, "Caucasian":1}, inplace=True)
    #Y.rename(columns={'readmit_30_days': 'label'}, inplace=True)
    print(Y)
    Y.columns = ['label']
    Y['label'].replace({True:1, False:0}, inplace=True)
    print(A)
    print(Y)
    final_df = pd.concat([df, A, Y], axis=1)
    #final_df = np.concatenate([df, A, Y], axis=1)

    print(final_df)

    # mapper = DataFrameMapper([(df.columns, StandardScaler())])
    # scaled_features = mapper.fit_transform(df.copy(), 4)
    # df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

    #scaler = StandardScaler().fit(df)
    #df = scaler.transform(df)
    #pipe.fit(df)
    #df = pipe.fit_transform(df)
    #print(df)
    #print(df["race"].value_counts())
    #print(df["readmit_30_days"].value_counts())

    #scaler = StandardScaler()
    #ohe = OneHotEncoder(sparse=False)

    # Scale and Encode Separate Columns
    #scaled_columns  = scaler.fit_transform(df[df.select_dtypes('number').columns.tolist()]) 
    #encoded_columns = ohe.fit_transform(df[categorical_features])

    # Concatenate (Column-Bind) Processed Columns Back Together
    #final_df = np.concatenate([scaled_columns, encoded_columns], axis=1)

    final_df.to_csv("dataset_ref.csv", index=False)

if __name__ == "__main__":
    main()