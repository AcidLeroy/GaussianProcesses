function [ my_fig, error ]  = prob3_4c()
%code for Non-linear Gaussian Process for a model for whichthe input
%contains only 5 values of w[n?1] and the output is corrupted by only white noise 
clear all; close all; clc;
N = 200;
nTrain = 100;
nTest = N-nTrain;
mean = 0;
var = 1;
w = mean + var.*randn(N,1);
B = [1,-2.3695,2.3140,-1.0547,0.1874];
A = [0.0048,0.0193,0.0289,0.0193,0.0048];
output = filter(A,B,w);

r=randperm(N);
index=r(1:nTrain);

diffI = setdiff(1:N,index);
%  w = 1:100;
w1 = [w(1:end)'];
w2 = [0 w(1:end-1)'];
w3 = [0 0 w(1:end-2)'];
w4 = [0 0 0 w(1:end-3)'];
w5 = [0 0 0 0 w(1:end-4)'];
W = [w1;w2;w3;w4;w5];
% 
% f1 = [0 output(1:end-1)'];
% f2 = [0 0 output(1:end-2)'];
% f3 = [0 0 0 output(1:end-3)'];
% f4 = [0 0 0 0 output(1:end-4)'];
% f5 = [0 0 0 0 0 output(1:end-5)'];
% F = [f1;f2;f3;f4;f5];
F=zeros(5,200);

X = [F;W];
x_train = X(:,index)';
x_test = X(:,diffI)';
Y = output;
y_train = Y(index);
y_test = Y(diffI);

meanfunc = @meanLinear; hyp.mean = zeros(10,1);
covfunc = {@covSum, {@covSEiso, @covNoise}}; hyp.cov = [0; 0; 0];
likfunc = @likGauss; hyp.lik = log(0.1);
hyp = minimize(hyp, @gp, -1000, @infExact, meanfunc, covfunc, likfunc, x_train, y_train);
nlml = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x_train, y_train);
%%
my_fig = figure(2);
[m,s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x_train, y_train, x_test);
x = 1:N;
xtest = x(diffI)';
xtrain = x(index)';
fi = [m+2*sqrt(s2); flip(m-2*sqrt(s2),1)];
fill([xtest; flip(xtest,1)], fi, [.9,.9,.9])
hold on; plot(xtest, m, 'b'); plot(xtrain, y_train, '*'); plot(Y, 'k--');
hold off;
axis tight;
xlabel('x[n]');
ylabel('f[n]');
title('Prediction of ARMA process with added white noise');
legend('Variance Margin','Predicted Output','Training Data', 'Real Values');
error = sum((m-y_test).^2)/length(m);

end

