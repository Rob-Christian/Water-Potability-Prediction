## Sensitivity and Probability of Artificial Neural Networks (2 layers)

# Creates empty array for storage
NN2_sensitivity_fold = np.zeros((1, 9, X.shape[0]))
NN2_sensitivity = np.zeros((1, 9, 10))
NN2_mean_sensitivity = np.zeros((1, 9))

for i1 in range(10):
    saved_model = load_model(r'C:\Users\...\Architecture1\a{}.h5'.format(10*3+i1+1))
    model2 = keras.Sequential([keras.layers.Dense(50, input_shape=(9,), weights=saved_model.layers[0].get_weights(), activation = 'relu')])
    
    # Computes for sensitivity using backpropagation
    for i2 in range(X.shape[0]):
        dout = sigmoid_backward(saved_model.predict(X.iloc[i2:i2+1]))
        dout = affine_backward(dout, (X.iloc[i2:i2+1], saved_model.get_weights()[2]))
        dout = relu_backward(dout, model2.predict(X.iloc[i2:i2+1]))
        dout = affine_backward(dout, (X.iloc[i2:i2+1], saved_model.get_weights()[0]))
        NN2_sensitivity_fold[:,:,i2] = dout
    
    # Gets the mean senstivity per fold
    NN2_sensitivity[:,:,i1] = np.mean(NN2_sensitivity_fold, axis = 2)
    NN2_sensitivity_fold = np.zeros((1, 9, X.shape[0]))

# Gets the overall mean sensitivity
NN2_mean_sensitivity = np.mean(NN2_sensitivity, axis = 2)
df1 = pd.DataFrame(NN2_mean_sensitivity.T)
df1.to_excel(r'C:\Users\...\Binary Sensitivities ANN2.xlsx', index = False)
