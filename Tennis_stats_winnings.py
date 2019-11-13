import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
players = pd.read_csv('tennis_stats.csv')
# print(players.head())
# print(players.columns)
# print(players.describe())


# perform exploratory analysis here:
plt.scatter(players['BreakPointsOpportunities'], players['Winnings'])
plt.title('Break Points Opportunities vs Winnings')
plt.xlabel('Break Points Opportunities')
plt.ylabel('Winnings')
plt.show()
plt.clf()

plt.scatter(players['FirstServeReturnPointsWon'], players['Winnings'])
plt.title('FSRPW vs Winnings')
plt.xlabel('First Serve Return Points Won')
plt.ylabel('Winnings')
plt.show()
plt.clf()


## perform single feature linear regressions here:
features = players[['BreakPointsOpportunities']]
winnings = players[['Winnings']]
# Split data into training and test sets.
features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)
# Create Linear Regression model and train it on training data.
model = LinearRegression()
model.fit(features_train, winnings_train)
# Score model using test data.
print("Predicted Winnings with Break Point Opportunities test score:", model.score(features_test, winnings_test))
# Find predicted outcome based on the model and plot it against actual outcome.
prediction = model.predict(features_test)
plt.scatter(winnings_test, prediction, alpha=0.4)
plt.title("Break Points Opportunities vs Winnings")
plt.xlabel("Break Points Opportunities")
plt.ylabel("Winnings")
plt.show()
plt.clf()

features = players[['FirstServeReturnPointsWon']]
winnings = players[['Winnings']]
features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)
model = LinearRegression()
model.fit(features_train, winnings_train)
print("Predicted Winnings with First Serve Return Points Won test score:", model.score(features_test, winnings_test))
prediction = model.predict(features_test)
plt.scatter(winnings_test, prediction, alpha=0.4)
plt.title("First Serve Return Points Won vs Winnings")
plt.xlabel("First Serve Return Points Won")
plt.ylabel("Winnings")
plt.show()
plt.clf()

features = players[['FirstServePointsWon']]
winnings = players[['Winnings']]
features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)
model = LinearRegression()
model.fit(features_train, winnings_train)
print("Predicted Winnings with First Serve Return Points Won test score:", model.score(features_test, winnings_test))
prediction = model.predict(features_test)
plt.scatter(winnings_test, prediction, alpha=0.4)
plt.title("First Serve Points Won vs Winnings")
plt.xlabel("First Serve Points Won")
plt.ylabel("Winnings")
plt.show()
plt.clf()


## perform two feature linear regressions here:
features = players[['BreakPointsOpportunities', 'FirstServeReturnPointsWon']]
winnings = players[['Winnings']]
features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)
model = LinearRegression()
model.fit(features_train, winnings_train)
print("Predicted Winnings with 2 features test score:", model.score(features_test, winnings_test))
prediction = model.predict(features_test)
plt.scatter(winnings_test, prediction, alpha=0.4)
plt.title("Predicted Winnings vs Actual Winnings")
plt.xlabel("Actual Winnings")
plt.ylabel("Predicted Winnings")
plt.show()
plt.clf()



## perform multiple feature linear regressions here:
features = players[['BreakPointsOpportunities', 'BreakPointsSaved', 'ReturnPointsWon', 'Aces']]
winnings = players[['Winnings']]
features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)
model = LinearRegression()
model.fit(features_train, winnings_train)
print("Predicted Winnings with multiple features test score:", model.score(features_test, winnings_test))
prediction = model.predict(features_test)
plt.scatter(winnings_test, prediction, alpha=0.4)
plt.title("Predicted Winnings vs Actual Winnings")
plt.xlabel("Actual Winnings")
plt.ylabel("Predicted Winnings")
plt.show()
plt.clf()