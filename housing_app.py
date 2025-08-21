import pandas as pd
import seaborn as sns
from category_encoders import OneHotEncoder
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from glob import glob
import matplotlib.pyplot as plt


# Create wrangle function
def wrangle(file_path):
    #read csv file
    df= pd.read_csv(file_path)

    #subset data: apartment in Distrito Federal, price < 100000
    mask_apt = df["property_type"]=="apartment"
    mask_place = df["place_with_parent_names"].str.split("|", expand=True)[2] == "Distrito Federal"
    mask_price = df["price_aprox_usd"] < 100000
    mask= (mask_apt) & (mask_place) & (mask_price)
    df = df[mask]

    # trimming the bottom and top 10% of "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1,0.9])
    df= df[ df["surface_covered_in_m2"].between(low,high) ]

    # separate "lat" and "lon" columns
    df[["lat","lon"]]  =df["lat-lon"].str.split(",", expand=True).astype(float)

    #Create a "borough" feature from the "place_with_parent_names" column
    df["borough"] = df["place_with_parent_names"].str.split("|", expand=True)[1]

    #Drop columns that are more than 50% null values, containing low- or high-cardinality categorical values,  constitute leakage for the target,  multicollinearity.
    df.drop(columns= ["surface_total_in_m2","price_usd_per_m2","floor","rooms","expenses",
                          "lat-lon","place_with_parent_names","price","price_aprox_local_currency","price_per_m2","operation",
                         "currency","property_type","properati_url"], 
                inplace=True)
    return(df)


# Use glob to create the list files
files = sorted(glob("data/mexico-city-real-estate-*.csv"))

# append all csv files
df = []
for file in files:
    frame = wrangle(file)
    df.append(frame)

df = pd.concat(df,ignore_index = True)
print(df.info())
df.head()

#explore the distribution of the target variable
fig, ax = plt.subplots() 

# Plot the histogram on the axes object
ax.hist(df["price_aprox_usd"]) 

# Label axes using the axes 
ax.set_xlabel("Price [$]")
ax.set_ylabel("Count")
plt.show()


# Add title 
ax.set_title("Distribution of Apartment Prices")

# Split data into feature matrix `X_train` and target vector `y_train`.

X_train = df.drop(columns= "price_aprox_usd")
y_train =  df["price_aprox_usd"]

# basedline MAE
y_mean = y_train.mean()
y_pred_baseline = [y_mean]*len(y_train)
baseline_mae = mean_absolute_error(y_train,y_pred_baseline)
print("Mean apt price:", y_mean)
print("Baseline MAE:", baseline_mae)

# Build Model
model = make_pipeline(OneHotEncoder(use_cat_names=True),SimpleImputer(),Ridge() )
# Fit model
model.fit(X_train, y_train)

# Training performance
Y_pred_train = pd.Series(model.predict(X_train))
training_mae = mean_absolute_error(y_train,y_pred_baseline)
print("Training MAE:", training_mae)

# test and predict
X_test = pd.read_csv("data/mexico-city-test-features.csv")
print(X_test.info())
X_test.head()

y_test_pred = pd.Series(model.predict(X_test))
y_test_pred.head()

# communicate the results
coefficients = model.named_steps["ridge"].coef_
features = model.named_steps["onehotencoder"].get_feature_names()
feat_imp = pd.Series(coefficients, index= features).sort_values(key=abs, ascending= True)
feat_imp 

# plot most important features
fig, ax = plt.subplots(figsize=(8,6))

# Create the horizontal bar plot on the axes object
feat_imp.tail(10).plot(kind= "barh", ax=ax)

#  Label axes 
ax.set_xlabel("Importance [USD]") 
ax.set_ylabel("Feature")

# Add title 
ax.set_title("Feature Importances for Apartment Price")
plt.savefig("images/feature_importances.png", dpi=300, bbox_inches="tight")    
plt.show()