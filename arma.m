function [ y2] = arma( a, b, x )
%ARMA Generate an ARMA process

%coefficients -a(2:end)

y2=zeros(size(x));
X=zeros(size(b));
Y=zeros(1,length(a)-1);

%And we construct the recursion
for i=1:length(x)
    X=[X(2:end) x(i)];
    y2(i)=b*X'-a(2:end)*Y';
    Y=[y2(i) Y(1:end-1)];
end

end

