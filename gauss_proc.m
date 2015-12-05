function  gauss_proc( xtrain, xgen, xtest, seed)
%gauss_proc example code using the real gaussian process code. 
%    xtrain=sort(rand(1,20)*10-5);
%    xgen=sort(rand(1,10)*10-5);
%    xtest=-5:0.01:5;
   cov = {'covSum', {'covSEiso','covNoise'}};
   hypgen= [log(1) log(1)];
   rng(seed)
   ytrain=ones(size(xgen))*covSEiso(hypgen,xgen',xtrain')+.1*randn(size(xtrain));
   yreal=ones(size(xgen))*covSEiso(hypgen,xgen',xtest');

   %% regression
   hypSE= [log(1) log(1)];
   hypnoise = [log(sqrt(0.2))];
   hypconst=log(sqrt(1));
   Ktrain=covSEiso(hypSE,xtrain',xtrain')+... %SQUARE EXPONENTIAL
       covNoise(hypnoise,xtrain')+...         %NOISE
       covConst(hypconst,xtrain);       %CONSTANT
   C=pinv(Ktrain);
   alpha=C*ytrain';

   %% TEST
   Ktest=covSEiso(hypSE,xtrain',xtest')+...    %SQUARE EXPONENTIAL
       covConst(hypconst,xtrain',xtest');%CONSTANT
   % MEAN
   f=Ktest'*alpha; 
   % VARIANCE
   Cov1=covSEiso(hypSE,xtest',xtest')+covConst(hypconst,xtest',xtest');
   var=diag(Cov1)-diag(Ktest'*C*Ktest);

   %% REPRESENTATION
   % Confidence interval to 2 sigma (95%)
   fill([xtest fliplr(xtest)]',[f'+2*var' fliplr(f'-2*var')]',[1 0.9 0.9])
   hold all
   plot(xtest,f,'k')
   plot(xtrain,ytrain,'*')
   plot(xtest,yreal,'k--')


end

