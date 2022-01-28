"""
see explanations of this actions in data_wrangling notebook
"""

import pandas as pd
import numpy as np

# alone - flag show did passenger be alone or not

train_df['Alone'] = train_df[['SibSp', 'Parch']].apply(lambda p: 0 if (p[0] + p[1] != 0) else 1, axis=1)

# familiars - the general number of familiars people

train_df['Familiars'] = train_df.SibSp + train_df.Parch

