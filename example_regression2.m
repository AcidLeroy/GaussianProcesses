close all
clear all
%% DATA GENERATION
randn('seed',3);
rand('seed',6);
xtrain=sort(rand(1,20)*10-5);
xgen=sort(rand(1,10)*10-5);
xtest=-5:0.01:5;
hypgen= [log(1) log(1)];
ytrain=ones(size(xgen))*covSEiso(hypgen,xgen',xtrain')+.1*randn(size(xtrain));
yreal=ones(size(xgen))*covSEiso(hypgen,xgen',xtest');

%% regression
noises=logspace(-2,0,10);
meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [1; 1];
covfunc={'covSum',{@covSEiso, @covConst}};
for i=1:length(noises)
    hypSE= [log(noises(i));log(1)];
    hypconst=log(sqrt(1));
    likfunc = @likGauss;
   
    hyp.cov=[hypSE;hypconst];
    hyp.lik=log(0.1);
    hyp2 = minimize(hyp, @gp, -100, @infEP, meanfunc, covfunc, likfunc, xtrain', ytrain');
    exp(hyp2.lik)
    [m s2 fmu fs2 ] = gp(hyp2, @infEP, meanfunc, covfunc, likfunc, xtrain', ytrain',xtest');
    
    %% REPRESENTATION
    % Confidence interval to 2 sigma (95%)
    figure(1)
    f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
    fill([xtest'; flipdim(xtest',1)], f, [1 0.9 0.9])
    hold on; plot(xtest, m); plot(xtrain, ytrain, '+')
    plot(xtest,yreal,'k--')
    hold off
    pause(1)
end