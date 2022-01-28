"""
see explanations of this actions in data_wrangling notebook
"""

from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_df)

scaled_train = pd.DataFrame(
    scaled_train, columns=train_df.columns, index=train_df.index
)
