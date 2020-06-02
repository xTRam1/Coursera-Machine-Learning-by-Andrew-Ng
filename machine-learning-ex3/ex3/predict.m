function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
X = [ones(m,1), X];

a_2 = sigmoid(X * Theta1');
a_3 = sigmoid([ones(size(a_2, 1), 1), a_2] * Theta2');
[value, p] = max(a_3, [], index=2)

end
