function  gauss_proc( xtrain, ytrain, yreal, xtest)
%gauss_proc example code using the real gaussian process code. 
%    xtrain=sort(rand(1,20)*10-5);
%    xgen=sort(rand(1,10)*10-5);
%    xtest=-5:0.01:5;
%    cov = {'covSum', {'conv_func','covNoise'}};
%    hypgen= [log(1) log(1)];
%    rng(seed)
%    ytrain=ones(size(xgen))*conv_func(hypgen,xgen',xtrain')+.1*randn(size(xtrain));
%    yreal=ones(size(xgen))*conv_func(hypgen,xgen',xtest');

%kernel_function = @(x1, x2) x1' * x2; 

   conv_func = @covSEiso; 

   %% regression
   hypSE= [log(1) log(1)];
   hypnoise = [log(sqrt(0.2))];
   hypconst=log(sqrt(1));
   
   Ktrain=conv_func(hypSE,xtrain',xtrain')+... %SQUARE EXPONENTIAL
       covNoise(hypnoise,xtrain')+...         %NOISE
       covConst(hypconst,xtrain);       %CONSTANT
   C=pinv(Ktrain);
   alpha=C*ytrain';

   %% TEST
   Ktest=conv_func(hypSE,xtrain',xtest')+...    %SQUARE EXPONENTIAL
       covConst(hypconst,xtrain',xtest');%CONSTANT
   % MEAN
   f=Ktest'*alpha; 
   % VARIANCE
   Cov1=conv_func(hypSE,xtest',xtest')+covConst(hypconst,xtest',xtest');
   var=diag(Cov1)-diag(Ktest'*C*Ktest);

   %% REPRESENTATION
   % Confidence interval to 2 sigma (95%)
   sigma = .1;
   fill([xtest fliplr(xtest)]',[f'+sigma*var' fliplr(f'-sigma*var')]',[.8 0.8 0.8])
   hold all
   plot(xtest,f,'b')
   plot(xtrain,ytrain,'*')
   plot(xtest,yreal,'k--')
   s = sprintf('confidence with \\sigma = %.2f', sigma);
   legend(s,'mean (predicted)', 'training data', 'real values')
   xlabel('input')
   ylabel('output')


end

