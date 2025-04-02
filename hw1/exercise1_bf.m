%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Starter code for Exercise 1 in hw1
%
% Last updated: $Date: 2022/10/6 $
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Generate data
clear;
close all;

% parameters
N = 15; % number of cells
n0 = 10; % index of initial cell 

% initialize belief
bel = zeros(N,1);
bel(n0) = 1;

for ii = 1:5 % move forward for five times
    bel = BayesFilter(bel,1);
end

for ii = 1:3 % move backward for three times
    bel = BayesFilter(bel,-1);
end

plot_belief(bel);

%% helper functions
function bel = BayesFilter(bel,d)
    % arguments: bel -- belief, i.e., probability distribution of robot's position
    %            d   -- move command. 1=forward, -1=backward
    % return:    bel -- updated belief
    N = length(bel);
    new_bel = zeros(size(bel));

    if d == 1
        % Forward motion
        for i = 1:N
            % Stay in place (25%)
            new_bel(i) = new_bel(i) + 0.25 * bel(i);
            
            % Move one cell forward (50%)
            if i < N
                new_bel(i+1) = new_bel(i+1) + 0.50 * bel(i);
            else
                % At last cell, stay in place
                new_bel(i) = new_bel(i) + 0.50 * bel(i);
            end
            
            % Move two cells forward (25%)
            if i < N-1
                new_bel(i+2) = new_bel(i+2) + 0.25 * bel(i);
            elseif i == N-1
                % At second-to-last cell, can only move one forward
                new_bel(N) = new_bel(N) + 0.25 * bel(i);
            else
                % At last cell, stay in place
                new_bel(i) = new_bel(i) + 0.25 * bel(i);
            end
        end

    elseif d == -1
        % Backward motion (mirror of forward motion)
        for i = 1:N
            % Stay in place (25%)
            new_bel(i) = new_bel(i) + 0.25 * bel(i);
            
            % Move one cell backward (50%)
            if i > 1
                new_bel(i-1) = new_bel(i-1) + 0.50 * bel(i);
            else
                % At first cell, stay in place
                new_bel(i) = new_bel(i) + 0.50 * bel(i);
            end
            
            % Move two cells backward (25%)
            if i > 2
                new_bel(i-2) = new_bel(i-2) + 0.25 * bel(i);
            elseif i == 2
                % At second cell, can only move one backward
                new_bel(1) = new_bel(1) + 0.25 * bel(i);
            else
                % At first cell, stay in place
                new_bel(i) = new_bel(i) + 0.25 * bel(i);
            end
        end

    else
        warning('Move command cannot be recognized')
    end
    
    bel = new_bel;
end

% plot belief
function plot_belief(bel)
% arguments: bel -- belief, i.e., probability distribution of robot's position
    len = length(bel);
    histogram('Categories',string(1:len),'BinCounts',bel);
    title('\bf Belief of Robot`s Position'); 
    xlabel('Time step'); 
    ylabel('Probability');
end