## Feature Engineering

# Delete rows with NaN values
X_raw = pd.read_csv(r"C:\Users\rmcad\Downloads\Research\Water Potability\water_potability.csv")
X_whole = X_raw.dropna(axis = 0)
X_retrieved = X_raw.dropna(axis = 0)
X_retrieved.reset_index(drop = True, inplace = True)

# Good range for PH values
IQR_ph = np.percentile(X_retrieved['ph'], 75) - np.percentile(X_retrieved['ph'], 25)
upper_ph = np.percentile(X_retrieved['ph'], 75) + 1.5*IQR_ph
lower_ph = np.percentile(X_retrieved['ph'], 25) - 1.5*IQR_ph

# Good range for Hardness values
IQR_hard = np.percentile(X_retrieved['Hardness'], 75) - np.percentile(X_retrieved['Hardness'], 25)
upper_hard = np.percentile(X_retrieved['Hardness'], 75) + 1.5*IQR_hard
lower_hard = np.percentile(X_retrieved['Hardness'], 25) - 1.5*IQR_hard

# Good range for Solids values
IQR_solids = np.percentile(X_retrieved['Solids'], 75) - np.percentile(X_retrieved['Solids'], 25)
upper_solids = np.percentile(X_retrieved['Solids'], 75) + 1.5*IQR_solids
lower_solids = np.percentile(X_retrieved['Solids'], 25) - 1.5*IQR_solids

# Good range for Sulfate values
IQR_sulfate = np.percentile(X_retrieved['Sulfate'], 75) - np.percentile(X_retrieved['Sulfate'], 25)
upper_sulfate = np.percentile(X_retrieved['Sulfate'], 75) + 1.5*IQR_sulfate
lower_sulfate = np.percentile(X_retrieved['Sulfate'], 25) - 1.5*IQR_sulfate

# Good range for Conductivity values
IQR_cond = np.percentile(X_retrieved['Conductivity'], 75) - np.percentile(X_retrieved['Conductivity'], 25)
upper_cond = np.percentile(X_retrieved['Conductivity'], 75) + 1.5*IQR_cond
lower_cond = np.percentile(X_retrieved['Conductivity'], 25) - 1.5*IQR_cond

# Good range for Organic Carbon values
IQR_org = np.percentile(X_retrieved['Organic_carbon'], 75) - np.percentile(X_retrieved['Organic_carbon'], 25)
upper_org = np.percentile(X_retrieved['Organic_carbon'], 75) + 1.5*IQR_org
lower_org = np.percentile(X_retrieved['Organic_carbon'], 25) - 1.5*IQR_org

# Good range for Trihalomethanes values
IQR_tri = np.percentile(X_retrieved['Trihalomethanes'], 75) - np.percentile(X_retrieved['Trihalomethanes'], 25)
upper_tri = np.percentile(X_retrieved['Trihalomethanes'], 75) + 1.5*IQR_tri
lower_tri = np.percentile(X_retrieved['Trihalomethanes'], 25) - 1.5*IQR_tri

# Good range for Turbidity values
IQR_tb = np.percentile(X_retrieved['Turbidity'], 75) - np.percentile(X_retrieved['Turbidity'], 25)
upper_tb = np.percentile(X_retrieved['Turbidity'], 75) + 1.5*IQR_tb
lower_tb = np.percentile(X_retrieved['Turbidity'], 25) - 1.5*IQR_tb

X = X_retrieved[(X_retrieved['ph'] >= lower_ph) & (X_retrieved['ph'] <= upper_ph) &
               (X_retrieved['Hardness'] >= lower_hard) & (X_retrieved['Hardness'] <= upper_hard) &
            (X_retrieved['Solids'] >= lower_solids) & (X_retrieved['Solids'] <= upper_solids) &
            (X_retrieved['Sulfate'] >= lower_sulfate) & (X_retrieved['Sulfate'] <= upper_sulfate) &
            (X_retrieved['Conductivity'] >= lower_cond) & (X_retrieved['Conductivity'] <= upper_cond) &
            (X_retrieved['Organic_carbon'] >= lower_org) & (X_retrieved['Organic_carbon'] <= upper_org) &
            (X_retrieved['Trihalomethanes'] >= lower_tri) & (X_retrieved['Trihalomethanes'] <= upper_tri) &
            (X_retrieved['Turbidity'] >= lower_tb) & (X_retrieved['Turbidity'] <= upper_tb)]

X_retrieved = X_retrieved[(X_retrieved['ph'] >= lower_ph) & (X_retrieved['ph'] <= upper_ph) &
               (X_retrieved['Hardness'] >= lower_hard) & (X_retrieved['Hardness'] <= upper_hard) &
            (X_retrieved['Solids'] >= lower_solids) & (X_retrieved['Solids'] <= upper_solids) &
            (X_retrieved['Sulfate'] >= lower_sulfate) & (X_retrieved['Sulfate'] <= upper_sulfate) &
            (X_retrieved['Conductivity'] >= lower_cond) & (X_retrieved['Conductivity'] <= upper_cond) &
            (X_retrieved['Organic_carbon'] >= lower_org) & (X_retrieved['Organic_carbon'] <= upper_org) &
            (X_retrieved['Trihalomethanes'] >= lower_tri) & (X_retrieved['Trihalomethanes'] <= upper_tri) &
            (X_retrieved['Turbidity'] >= lower_tb) & (X_retrieved['Turbidity'] <= upper_tb)]

# Drop potability in X
y = X['Potability'].to_numpy()
y = np.reshape(y, (X.shape[0], 1))
X.drop("Potability", axis = 1, inplace = True)
