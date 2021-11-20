import pandas as pd
import pickle

df = pd.read_csv('IPL_DATA.csv')

df['bat_team'].unique()

main_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']

df = df[(df['bat_team'].isin(main_teams)) & (df['bowl_team'].isin(main_teams))]

df = df[df['overs']>=5.0]

encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])

encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total' , 'Year']]

X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['Year']<= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['Year']>= 2017]

y_train = encoded_df[encoded_df['Year']<= 2016]['total'].values
y_test = encoded_df[encoded_df['Year']>= 2017]['total'].values

X_train.drop(labels='Year', axis=True, inplace=True)
X_test.drop(labels='Year', axis=True, inplace=True)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

filename = 'first-innings-score-prediction-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))
