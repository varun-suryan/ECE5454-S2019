clear all; close all; clc;

% normalize the dataset
X = normalize(csvread('datamatrix.csv', 1, 1)');

% Calculate the svd here
[U, S, ~] = svd((X'*X)/size(X, 1));

%project onto two dimensions
projected_data = X*U(:, 1:2);

% recover the original data again by projecting back
X_recovered = projected_data*U(:, 1:2)';

fprintf('Eigen vectors are:\n')
disp(U)

fprintf('Eigen values are:\n')
disp(diag(S))

