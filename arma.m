function [ x_t ] = arma( ma_coefs, ar_coefs, epsilon_t, c)
%ARMA Generate an ARMA process
x_t = zeros(size(epsilon_t)); 


for t = 1:(length(epsilon_t) - length(ma_coefs))
    x_t(length(epsilon_t) + 1 - t) = epsilon_t(length(epsilon_t)+1  - t) + c;
    for i = 1:length(ar_coefs)  % Auto regressive
        x_t(length(epsilon_t) + 1-t) = x_t(length(epsilon_t) + 1-t) - ...
           ar_coefs(i)*x_t(length(epsilon_t) + 2 - t - i);
    end
    for i = 1:length(ma_coefs) % Moving average
       x_t(length(epsilon_t) + 1-t) = x_t(length(epsilon_t) + 1-t) + ...
          ma_coefs(i)*epsilon_t(length(epsilon_t) + 2 - t - i);
    end
end

end

