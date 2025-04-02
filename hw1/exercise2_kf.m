%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Supplemental Matlab code for Exercise 2 in hw1
%
% Last updated: $Date: 2022/10/6 $
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Generate data
close all;

% Lock random seed
randn('state',123);

% Gaussian random draw, m is the mean and S the covariance
gauss_rnd = @(m,S) m + chol(S)'*randn(size(m));

% Calculate root mean squared error
rmse = @(x,y) sqrt(mean((x(:)-y(:)).^2));

% Define parameters
steps = 100;  % Number of time steps
w     = 0.5;  % Angular velocity
q     = 0.01; % Process noise spectral density
r     = 0.1;  % Measurement noise variance

% This is the transition matrix
A = [cos(w)    sin(w)/w; 
   -w*sin(w) cos(w)];

% This is the process noise covariance
Q = [0.5*q*(w-cos(w)*sin(w))/w^3 0.5*q*sin(w)^2/w^2;
   0.5*q*sin(w)^2/w^2          0.5*q*(w+cos(w)*sin(w))/w];

% This is the true initial value
x0 = [0;0.1]; 

% Simulate data
X = zeros(2,steps);  % The true signal
Y = zeros(1,steps);  % Measurements
T = 1:steps;         % Time
x = x0;
for k=1:steps
    x = gauss_rnd(A*x,Q);
    y = gauss_rnd(x(1),r);
    X(:,k) = x;
    Y(:,k) = y;
end

% Visualize
figure; clf;
plot(T,X(1,:),'--',T,Y,'o');
legend('True signal','Measurements');
xlabel('Time step'); title('\bf Simulated data')

%% Baseline solution

% Baseline solution. The estimates
% of x_k are stored as columns of
% the matrix EST1.

% Calculate baseline estimate
m1 = [0;1];  % Initialize first step with a guess
EST1 = zeros(2,steps);
for k=1:steps
    m1(2) = Y(k)-m1(1);
    m1(1) = Y(k);
    EST1(:,k) = m1;
end

% Visualize results
figure; clf;

% Plot the signal and its estimate
subplot(2,1,1);
plot(T,X(1,:),'--',T,EST1(1,:),'-',T,Y,'o');
legend('True signal','Estimated signal','Measurements');
xlabel('Time step'); title('\bf Baseline solution')

% Plot the derivative and its estimate
subplot(2,1,2);
plot(T,X(2,:),'--',T,EST1(2,:),'-');
legend('True derivative','Estimated derivative');
xlabel('Time step')

% Compute error
err1 = rmse(X,EST1)

%% Kalman filter

% Kalman filter solution. The estimates
% of x_k are stored as columns of
% the matrix EST2.

m2 = [0;1];  % Initialize first step
P2 = eye(2); % Some uncertanty in covariance  
EST2 = zeros(2,steps); % Allocate space for results

% Run Kalman filter
for k=1:steps
    %%% Replace this part with the Kalman filter equations
    %%% Kalman filter equations start here %%%
    
    % ----- Prediction step -----
    % Predict state
    m_pred = A * m2;
    % Predict covariance
    P_pred = A * P2 * A' + Q;
    
    % ----- Update step -----
    % Observation matrix
    H = [1, 0];
    % Measurement residual
    y_res = Y(k) - H * m_pred;
    % Residual covariance
    S = H * P_pred * H' + r;
    % Kalman gain
    K = P_pred * H' / S;  % For scalar S, division is more efficient than inv()
    
    % Update state estimate
    m2 = m_pred + K * y_res;
    % Update covariance estimate
    P2 = (eye(2) - K * H) * P_pred;
    
    %%% Kalman filter equations end here %%%

    % Store the results
    EST2(:,k) = m2;
end

% Visualize results
figure; clf

% Plot the signal and its estimate
subplot(2,1,1);
plot(T,X(1,:),'--',T,EST2(1,:),'-',T,Y,'o');
legend('True signal','Estimated signal','Measurements');
xlabel('Time step'); title('\bf Kalman filter')

% Plot the derivative and its estimate
subplot(2,1,2);
plot(T,X(2,:),'--',T,EST2(2,:),'-');
legend('True derivative','Estimated derivative');
xlabel('Time step')

% Compute error
err2 = rmse(X,EST2)