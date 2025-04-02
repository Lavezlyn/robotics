function search_ukf_params_advanced()
    % 第一阶段：粗略搜索
    fprintf('第一阶段：粗略搜索开始...\n');
    alpha_range_coarse = 0.1:0.2:2.0;      % 扩大alpha范围
    beta_range_coarse = [0, 1, 2, 3];      % 扩大beta范围
    kappa_range_coarse = -1:0.2:2.0;       % 扩大且包含负值的kappa范围
    
    [best_params_coarse, best_rmse_coarse] = grid_search(alpha_range_coarse, ...
        beta_range_coarse, kappa_range_coarse, '粗略搜索');
    
    % 第二阶段：在最优点附近进行细粒度搜索
    fprintf('\n第二阶段：细粒度搜索开始...\n');
    alpha_min = max(0.1, best_params_coarse.alpha - 0.2);
    alpha_max = best_params_coarse.alpha + 0.2;
    alpha_range_fine = alpha_min:(0.02):alpha_max;
    
    % beta使用离最优beta最近的几个值
    best_beta_idx = find(beta_range_coarse == best_params_coarse.beta);
    beta_range_fine = [best_params_coarse.beta-0.5, best_params_coarse.beta, ...
        best_params_coarse.beta+0.5];
    
    kappa_min = best_params_coarse.kappa - 0.2;
    kappa_max = best_params_coarse.kappa + 0.2;
    kappa_range_fine = kappa_min:(0.02):kappa_max;
    
    [best_params_fine, best_rmse_fine] = grid_search(alpha_range_fine, ...
        beta_range_fine, kappa_range_fine, '细粒度搜索');
    
    % 输出最终结果
    fprintf('\n搜索完成！\n');
    fprintf('粗略搜索最优结果：\n');
    fprintf('alpha=%.3f, beta=%.3f, kappa=%.3f, RMSE=%.6f\n', ...
        best_params_coarse.alpha, best_params_coarse.beta, ...
        best_params_coarse.kappa, best_rmse_coarse);
    
    fprintf('\n细粒度搜索最优结果：\n');
    fprintf('alpha=%.3f, beta=%.3f, kappa=%.3f, RMSE=%.6f\n', ...
        best_params_fine.alpha, best_params_fine.beta, ...
        best_params_fine.kappa, best_rmse_fine);
    
    % 使用最优参数运行一次UKF并可视化结果
    visualize_best_result(best_params_fine);
end

function [best_params, best_rmse] = grid_search(alpha_range, beta_range, kappa_range, search_name)
    % 存储最优结果
    best_rmse = Inf;
    best_params = struct('alpha', 0, 'beta', 0, 'kappa', 0);
    results = [];
    
    % 生成仿真数据
    [X, Y, T] = generate_data();
    
    % 计算总迭代次数
    total_iterations = length(alpha_range) * length(beta_range) * length(kappa_range);
    current_iter = 0;
    
    % 创建进度条
    progress_bar = waitbar(0, sprintf('%s 进度...', search_name));
    
    try
        for alpha = alpha_range
            for beta = beta_range
                for kappa = kappa_range
                    current_iter = current_iter + 1;
                    
                    % 更新进度条
                    waitbar(current_iter/total_iterations, progress_bar, ...
                        sprintf('%s 进度: %.1f%%', search_name, 100*current_iter/total_iterations));
                    
                    % 运行UKF获取RMSE
                    try
                        rmse_value = run_ukf_with_params(X, Y, T, alpha, beta, kappa);
                        
                        % 保存结果
                        results = [results; alpha, beta, kappa, rmse_value];
                        
                        % 更新最优参数
                        if rmse_value < best_rmse
                            best_rmse = rmse_value;
                            best_params.alpha = alpha;
                            best_params.beta = beta;
                            best_params.kappa = kappa;
                            
                            fprintf('新的最优参数 (%s): alpha=%.3f, beta=%.3f, kappa=%.3f, RMSE=%.6f\n', ...
                                search_name, alpha, beta, kappa, rmse_value);
                        end
                    catch
                        fprintf('参数组合无效: alpha=%.3f, beta=%.3f, kappa=%.3f\n', ...
                            alpha, beta, kappa);
                        continue;
                    end
                end
            end
        end
    catch e
        delete(progress_bar);
        rethrow(e);
    end
    
    delete(progress_bar);
    
    % 可视化结果
    visualize_results(results, best_params, search_name);
end

function visualize_best_result(params)
    [X, Y, T] = generate_data();
    
    % 运行UKF
    steps = length(T);
    m2 = 0.8;
    P2 = 1;
    EST2 = zeros(1,steps);
    
    % 使用最优参数运行UKF
    [EST2, P_history] = run_ukf_with_params_full(X, Y, T, params.alpha, params.beta, params.kappa);
    
    % 创建新图形
    figure('Name', '最优UKF结果');
    
    % 绘制状态估计
    subplot(2,1,1);
    plot(T, X, '--', 'DisplayName', '真实状态');
    hold on;
    plot(T, EST2, '-', 'DisplayName', '估计状态');
    plot(T, Y, 'o', 'DisplayName', '测量值');
    
    % 添加3σ置信区间
    sigma = sqrt(squeeze(P_history));
    fill([T, fliplr(T)], ...
         [EST2+3*sigma, fliplr(EST2-3*sigma)], ...
         'g', 'FaceAlpha', 0.2, 'DisplayName', '3σ置信区间');
    
    legend('Location', 'best');
    title(sprintf('UKF结果 (α=%.3f, β=%.3f, κ=%.3f)', ...
        params.alpha, params.beta, params.kappa));
    xlabel('时间步');
    ylabel('状态');
    
    % 绘制误差
    subplot(2,1,2);
    error = X - EST2;
    plot(T, error, 'b-', 'DisplayName', '估计误差');
    hold on;
    plot(T, 3*sigma, 'r--', 'DisplayName', '3σ边界');
    plot(T, -3*sigma, 'r--', 'HandleVisibility', 'off');
    legend('Location', 'best');
    title('估计误差');
    xlabel('时间步');
    ylabel('误差');
    
    % 计算并显示性能指标
    rmse = sqrt(mean(error.^2));
    mae = mean(abs(error));
    
    fprintf('\n性能指标：\n');
    fprintf('RMSE: %.6f\n', rmse);
    fprintf('MAE: %.6f\n', mae);
    fprintf('平均3σ边界: %.6f\n', mean(3*sigma));
end

% 修改run_ukf_with_params函数以返回完整的协方差历史
function [EST2, P_history] = run_ukf_with_params_full(X, Y, T, alpha, beta, kappa)
    steps = length(T);
    m2 = 0.8;
    P2 = 1;
    EST2 = zeros(1,steps);
    P_history = zeros(1,steps);
    
    % 模型定义
    f = @(x) x-0.01*sin(x);
    h = @(x) 0.5*sin(2*x);
    q = 0.01^2;
    r = 0.02;
    
    n = 1;  % 状态维度
    lambda = alpha^2 * (n + kappa) - n;
    
    % 计算权重
    Wm = zeros(2*n + 1, 1);
    Wc = zeros(2*n + 1, 1);
    Wm(1) = lambda/(n + lambda);
    Wc(1) = lambda/(n + lambda) + (1 - alpha^2 + beta);
    for i = 2:(2*n + 1)
        Wm(i) = 1/(2*(n + lambda));
        Wc(i) = 1/(2*(n + lambda));
    end
    
    % UKF主循环
    for k=1:steps
        % Sigma点生成
        sP = sqrt((n + lambda) * P2);
        Xi = [m2, m2 + sP, m2 - sP];
        
        % 预测步骤
        fXi = zeros(1, 2*n + 1);
        for i = 1:(2*n + 1)
            fXi(i) = f(Xi(i));
        end
        
        % 预测均值和协方差
        m2_pred = 0;
        for i = 1:(2*n + 1)
            m2_pred = m2_pred + Wm(i) * fXi(i);
        end
        
        P2_pred = q;
        for i = 1:(2*n + 1)
            P2_pred = P2_pred + Wc(i) * (fXi(i) - m2_pred) * (fXi(i) - m2_pred)';
        end
        
        % 更新步骤
        Yi = zeros(1, 2*n + 1);
        for i = 1:(2*n + 1)
            Yi(i) = h(fXi(i));
        end
        
        y_pred = 0;
        for i = 1:(2*n + 1)
            y_pred = y_pred + Wm(i) * Yi(i);
        end
        
        Pyy = r;
        for i = 1:(2*n + 1)
            Pyy = Pyy + Wc(i) * (Yi(i) - y_pred) * (Yi(i) - y_pred)';
        end
        
        Pxy = 0;
        for i = 1:(2*n + 1)
            Pxy = Pxy + Wc(i) * (fXi(i) - m2_pred) * (Yi(i) - y_pred)';
        end
        
        % Kalman增益和状态更新
        K = Pxy / Pyy;
        m2 = m2_pred + K * (Y(k) - y_pred);
        P2 = P2_pred - K * Pyy * K';
        
        % 存储结果
        EST2(k) = m2;
        P_history(k) = P2;
    end
end

function [X, Y, T] = generate_data()
    % 锁定随机种子
    randn('state',101);
    
    % 参数设置
    steps = 100;
    q = 0.01^2;
    r = 0.02;
    
    % 模型定义
    f = @(x) x-0.01*sin(x);
    h = @(x) 0.5*sin(2*x);
    gauss_rnd = @(m,S) m + chol(S)'*randn(size(m));
    
    % 生成数据
    X = zeros(1,steps);
    Y = zeros(1,steps);
    T = 1:steps;
    x = 1; % 初始状态
    
    for k=1:steps
        x = gauss_rnd(f(x),q);
        y = gauss_rnd(h(x),r);
        X(k) = x;
        Y(k) = y;
    end
end

function rmse_value = run_ukf_with_params(X, Y, T, alpha, beta, kappa)
    try
        % 检查参数有效性
        if alpha <= 0 || beta < 0 || alpha^2 * (1 + kappa) <= 0
            rmse_value = NaN;
            return;
        end
        
        [EST2, ~] = run_ukf_with_params_full(X, Y, T, alpha, beta, kappa);
        
        % 计算RMSE
        rmse_value = sqrt(mean((X(:)-EST2(:)).^2));
        
        % 检查结果有效性
        if isnan(rmse_value) || ~isreal(rmse_value)
            rmse_value = NaN;
        end
    catch
        rmse_value = NaN;
    end
end

function visualize_results(results, best_params, search_name)
    figure('Name', sprintf('%s 结果', search_name));
    
    % 创建唯一的alpha-kappa对
    unique_ak = unique(results(:,[1,3]), 'rows');
    
    % 对每个beta值绘制一个子图
    unique_beta = unique(results(:,2));
    num_subplots = length(unique_beta);
    
    for i = 1:num_subplots
        subplot(1, num_subplots, i);
        beta_idx = results(:,2) == unique_beta(i);
        beta_results = results(beta_idx, :);
        
        % 创建网格数据
        [X, Y] = meshgrid(unique(beta_results(:,1)), unique(beta_results(:,3)));
        Z = griddata(beta_results(:,1), beta_results(:,3), beta_results(:,4), X, Y);
        
        % 绘制等高线图
        contourf(X, Y, Z, 20);
        colorbar;
        hold on;
        
        % 如果这是最优beta，标记最优点
        if unique_beta(i) == best_params.beta
            plot(best_params.alpha, best_params.kappa, 'r*', 'MarkerSize', 10);
        end
        
        title(sprintf('beta = %.1f', unique_beta(i)));
        xlabel('alpha');
        ylabel('kappa');
    end
    
    sgtitle(sprintf('%s 结果 (RMSE)', search_name));
end