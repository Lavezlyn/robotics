%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Supplemental Matlab code for Exercise 4 in hw1
%
% Last updated: $Date: 2022/10/6 $
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Generate data
clear;
clc;
close all;

% Lock random seed
randn('state',101);

% Gaussian random draw, m is the mean and S the covariance
gauss_rnd = @(m,S) m + chol(S)'*randn(size(m));

% Calculate root mean squared error
rmse = @(x,y) sqrt(mean((x(:)-y(:)).^2));

% Define parameters
steps = 100;  % Number of time steps
% w     = 0.5;  % Angular velocity
q     = 0.01^2; % Process noise spectral density
r     = 0.02;  % Measurement noise variance

% This is the dynamics model
f = @(x) x-0.01*sin(x);

% This is the measurement model
h = @(x) 0.5*sin(2*x);

% This is the true initial value
x0 = 1; 

% Simulate data
X = zeros(1,steps);  % The true state
Y = zeros(1,steps);  % Measurements
T = 1:steps;         % Time
x = x0;
for k=1:steps
    x = gauss_rnd(f(x),q);
    y = gauss_rnd(h(x),r);
    X(k) = x;
    Y(k) = y;
end

% Visualize
figure; clf;
plot(T,X(1,:),'--',T,Y,'o');
legend('True signal','Measurements');
xlabel('Time step'); title('\bf Simulated data')

%% Extended Kalman filter

% Extended Kalman filter solution. The estimates
% of x_t are stored in EST2.

m = 0.8;  % Initialize first step
P = 1; % Variance  
EST = zeros(1,steps); % Allocate space for results

% Run EKF
for k=1:steps
    % Prediction step
    % Calculate Jacobian of f(x)
    F = 1 - 0.01*cos(m);              % df/dx
    
    % Predict
    m_pred = f(m);                     % 预测状态
    P_pred = F * P * F' + q;          % 预测协方差
    
    % Update step
    % Calculate Jacobian of h(x)
    H = cos(2*m_pred);                % dh/dx
    
    % Kalman gain
    K = P_pred * H' / (H * P_pred * H' + r);
    
    % Update
    m = m_pred + K * (Y(k) - h(m_pred));
    P = (1 - K * H) * P_pred;
    
    % Store the results
    EST(k) = m;
end

% Visualize results
figure; clf

% Plot the state and its estimate
plot(T,X(:),'--',T,EST(:),'-',T,Y,'o');
legend('True state','Estimated state','Measurements');
xlabel('Time step'); title('\bf Extended Kalman filter')

% Compute error
err = rmse(X,EST)

%% Unscented Kalman filter
% Unscented Kalman filter solution. The estimates
% of x_t are stored in EST3.

m2 = 0.8;  % Initialize first step
P2 = 1; % Variance  
EST2 = zeros(1,steps); % Allocate space for results

% Run UKF
for k=1:steps
    % UKF parameters
    alpha = 1.220;     % 扩展参数
    beta = 0.500;         % 高斯分布最优值
    kappa = -0.460;        % 次要扩展参数
    n = 1;           % 状态维度
    lambda = alpha^2 * (n + kappa) - n;
    
    % 计算权重
    Wm = zeros(2*n + 1, 1);    % 均值权重
    Wc = zeros(2*n + 1, 1);    % 协方差权重
    Wm(1) = lambda/(n + lambda);
    Wc(1) = lambda/(n + lambda) + (1 - alpha^2 + beta);
    for i = 2:(2*n + 1)
        Wm(i) = 1/(2*(n + lambda));
        Wc(i) = 1/(2*(n + lambda));
    end
    
    % Generate sigma points
    sP = sqrt((n + lambda) * P2);
    Xi = [m2, m2 + sP, m2 - sP];  % Sigma points
    
    % Prediction step
    % Transform sigma points through process model
    fXi = zeros(1, 2*n + 1);
    for i = 1:(2*n + 1)
        fXi(i) = f(Xi(i));
    end
    
    % Predicted mean and covariance
    m2_pred = 0;
    for i = 1:(2*n + 1)
        m2_pred = m2_pred + Wm(i) * fXi(i);
    end
    
    P2_pred = q;
    for i = 1:(2*n + 1)
        P2_pred = P2_pred + Wc(i) * (fXi(i) - m2_pred) * (fXi(i) - m2_pred)';
    end
    
    % Update step
    % Transform sigma points through measurement model
    Yi = zeros(1, 2*n + 1);
    for i = 1:(2*n + 1)
        Yi(i) = h(fXi(i));
    end
    
    % Predicted measurement
    y_pred = 0;
    for i = 1:(2*n + 1)
        y_pred = y_pred + Wm(i) * Yi(i);
    end
    
    % Innovation covariance
    Pyy = r;
    for i = 1:(2*n + 1)
        Pyy = Pyy + Wc(i) * (Yi(i) - y_pred) * (Yi(i) - y_pred)';
    end
    
    % Cross covariance
    Pxy = 0;
    for i = 1:(2*n + 1)
        Pxy = Pxy + Wc(i) * (fXi(i) - m2_pred) * (Yi(i) - y_pred)';
    end
    
    % Kalman gain and update
    K = Pxy / Pyy;
    m2 = m2_pred + K * (Y(k) - y_pred);
    P2 = P2_pred - K * Pyy * K';
    
    % Store the results
    EST2(k) = m2;
end

% Visualize results
figure; clf

% Plot the state and its estimate
plot(T,X(:),'--',T,EST2(:),'-',T,Y,'o');
legend('True state','Estimated state','Measurements');
xlabel('Time step'); title('\bf Unscented Kalman filter')

% Compute error
err2 = rmse(X,EST2)
