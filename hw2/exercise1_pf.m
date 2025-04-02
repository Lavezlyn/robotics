%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Supplemental Matlab code for Exercise 1 in hw2
%
% Last updated: $Date: 2023/11/5 $
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Generate data
clear;
clc;
close all;

% Lock random seed
rng(101,'twister');

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

%% Particle filter

% Particle filter solution. The estimates of x_t are stored in EST.
% Initialize the particles
particle_num = 100;
particles = 0.8*ones(particle_num,1);
weights = ones(particle_num,1)*1/particle_num;
EST = zeros(1,steps); % Allocate space for results

% Run EKF
for k=1:steps
    % Prediction step: propagate particles through the state transition
    particles = f(particles) + sqrt(q)*randn(particle_num,1);
    
    % Update step: calculate importance weights
    for i=1:particle_num
        weights(i) = exp(-0.5*(Y(k) - h(particles(i)))^2/r);
    end
    weights = weights/sum(weights); % Normalize weights
    
    % Resampling step: systematic resampling
    cumsum_weights = cumsum(weights);
    u = (0:particle_num-1)'/particle_num + rand/particle_num;
    j = 1;
    particles_new = zeros(particle_num,1);  % Pre-allocate the array
    for i=1:particle_num
        while u(i) > cumsum_weights(j)
            j = j + 1;
        end
        particles_new(i) = particles(j);
    end
    particles = particles_new;
    weights = ones(particle_num,1)/particle_num; % Reset weights
    
    % Store the mean value of particles
    EST(k) = mean(particles);
end

% Visualize results
figure; clf

% Plot the state and its estimate
plot(T,X(:),'--',T,EST(:),'-',T,Y(:),'o');
legend('True state','Estimated state','Measurements');
xlabel('Time step'); title('\bf Particle filter')

% Compute error
err = rmse(X,EST)