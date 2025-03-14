import pandas as pd

def clean_data(df):

    cols_to_drop = ['company', 'agent']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    df.dropna(inplace=True)
    
    if 'adr' in df.columns:
        df = df[(df['adr'] > 0) & (df['adr'] < 5000)]
    
    return df

df = pd.read_csv("/Users/lasserathke/Desktop/Universität/Python/Hotel Dataset/hotel_booking.csv")
df_cleaned = clean_data(df)
df_cleaned.to_csv("/Users/lasserathke/Desktop/Universität/Python/Hotel Dataset/cleaned_dataset.csv", index=False)
