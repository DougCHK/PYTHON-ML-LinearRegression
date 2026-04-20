import pandas
import matplotlib.pyplot
import sklearn.linear_model
# Y =
data = pandas.read_csv("cost_revenue_clean.csv")

X = pandas.DataFrame(data, columns=['production_budget_usd'])
y = pandas.DataFrame(data, columns=['worldwide_gross_usd'])
print(type(X))
regression = sklearn.linear_model.LinearRegression()
regression.fit(X, y)
print(regression.coef_) #theta 1
print(regression.intercept_) #theta 0
print(regression.predict(X))

matplotlib.pyplot.figure(figsize=(10,6))
matplotlib.pyplot.title("Film Cost vs Global Revenue")
matplotlib.pyplot.xlabel("Production Budget $")
matplotlib.pyplot.ylabel("Worldwide Gross $")
matplotlib.pyplot.scatter(X, y, alpha=0.3)
matplotlib.pyplot.plot(X, regression.predict(X), color='red', linewidth=4)
matplotlib.pyplot.ylim(0,3000000000)
matplotlib.pyplot.xlim(0,450000000)
matplotlib.pyplot.show()