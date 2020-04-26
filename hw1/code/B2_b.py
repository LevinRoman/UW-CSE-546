#B2.b
#Fit the model with the best p
#Based on the plot, p_hat = 3000
p_hat = 3100
transformed_X_train, G, b = non_linear_transform(X_train, p_hat)
W_hat_p = train(transformed_X_train, Y_train, 1e-4)

#Predict for validation set:
transformed_X_test, _, _ = non_linear_transform(X_test, p_hat, G, b)#use the same G,b
labels_test_pred = predict(W_hat_p, transformed_X_test)
print('Test error with p_hat:', accuracy_error(labels_test_pred, labels_test))