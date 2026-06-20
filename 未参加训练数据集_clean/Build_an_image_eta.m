% ================= plot_four_combinations_eff_2x2.m =================
% 功能：读取 4 个 CSV 文件，绘制 质量流量(X) vs 等熵效率(Y)
% 布局：2行2列 (上二下二)，每个文件一张子图
% 对应关系：Stage 1=黑, Stage 2=红, Stage 3=蓝, Stage 4=绿
clearvars; close all; clc;

%% ================= 用户配置区 =================
% 1. 定义文件名列表 (配合 2x2 布局，这里列出4个典型文件)
file_names = {
     '1-2-80-compressor.csv'
     '1-3-80-compressor.csv'
     '1-4-80-compressor.csv'
     '2-3-80-compressor.csv'
     '2-4-80-compressor.csv'
     '3-4-80-compressor.csv'

};

% 2. 定义 X 轴变量名 (Mass Flow Rate)
target_x_candidates = {'gc_kg_s', 'gc', 'massflow', 'mass_flow', 'flow', 'm_dot'}; 

% 3. 输出图片名
out_png = 'four_combinations_efficiency_2x2.png';

% 4. 颜色配置 (对应 Stage 1-4)
stage_colors = [0 0 0; 1 0 0; 0 0 1; 0 0.5 0]; 

%% ================= 代码逻辑 =================
f = figure('Units', 'normalized', 'Position', [0.1 0.1 0.7 0.8]); % 调整比例适应2x2
% 【修改点1】布局改为 2行 2列
t_layout = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- 处理大标题：提取全局背压 ---
if ~isempty(file_names)
    first_fn = file_names{1};
    global_nums = regexp(first_fn, '\d+', 'match');
    if ~isempty(global_nums)
        global_back_pressure = global_nums{end-1}; % 假设最后一个数字是背压
    else
        global_back_pressure = '?';
    end
else
    global_back_pressure = '?';
end

% 设置大标题
main_title_str = sprintf('Isentropic Efficiency vs Mass Flow Rate (Inlet pressure: %s bar)', global_back_pressure);
title(t_layout, main_title_str, 'FontSize', 16);

% --- 循环处理每个文件 ---
for k = 1:numel(file_names)
    raw_fn = file_names{k};
    
    % --- 文件名后缀处理 ---
    if endsWith(raw_fn, '.csv', 'IgnoreCase', true)
        fn = raw_fn;
        name_for_parsing = raw_fn(1:end-4);
    else
        fn = [raw_fn, '.csv'];
        name_for_parsing = raw_fn;
    end
    
    % --- 提取数字作为子标题 ---
    nums = regexp(name_for_parsing, '\d+', 'match');
    
    if length(nums) >= 2
        % 拼接调速级数字 (排除最后一个背压数字)
        stages_str = strjoin(nums(1:end-2), ' & ');
        subplot_title = sprintf('Regulation: Stages %s', stages_str);
    else
        subplot_title = sprintf('Case %d', k);
    end
    
    % --- 绘图准备 ---
    nexttile; hold on; grid on; box on;
    title(subplot_title, 'FontSize', 11);
    
    if ~isfile(fn)
        text(0.5, 0.5, 'File Not Found', 'HorizontalAlignment', 'center', 'Color', 'r');
        warning('文件不存在: %s', fn);
        continue;
    end
    
    % --- 读取数据 ---
    opts = detectImportOptions(fn);
    opts.PreserveVariableNames = true; 
    T = readtable(fn, opts);
    lvar = lower(T.Properties.VariableNames); 
    
    % --- 匹配 X 轴 ---
    x_idx = [];
    for i = 1:length(target_x_candidates)
        candidate = lower(target_x_candidates{i});
        idx = find(strcmpi(lvar, candidate), 1);
        if isempty(idx), idx = find(contains(lvar, candidate), 1); end
        if ~isempty(idx), x_idx = idx; break; end
    end
    
    % --- 匹配 Y 轴 (效率: eta_s1, eff1...) ---
    eta_idxs = zeros(1,4);
    for s = 1:4
        patterns = {
            sprintf('eta_s%d',s), ...
            sprintf('eta%d',s), ...
            sprintf('eff%d',s), ...
            sprintf('stage%d',s)
        };
        for p = 1:numel(patterns)
            idx = find(contains(lvar, patterns{p}), 1);
            if ~isempty(idx), eta_idxs(s) = idx; break; end
        end
    end
    
    if isempty(x_idx) || any(eta_idxs == 0)
        text(0.5, 0.5, 'Data Missing', 'HorizontalAlignment', 'center', 'Color', 'r');
        continue;
    end
    
    % --- 提取并排序 ---
    X_Data = double(T{:, x_idx});
    Etas = nan(height(T), 4);
    for s = 1:4
        if eta_idxs(s) > 0
            Etas(:, s) = double(T{:, eta_idxs(s)});
        end
    end
    
    [X_Data, sort_idx] = sort(X_Data);
    Etas = Etas(sort_idx, :);
    
    % --- 绘制曲线 ---
    for level = 1:4
        if ~isnan(Etas(1, level))
            plot(X_Data, Etas(:, level), 'Color', stage_colors(level, :), ...
                 'LineWidth', 1.5, 'DisplayName', sprintf('Stage %d', level));
        end
    end
    
    % --- 【修改点2】坐标轴标签逻辑适应 2x2 ---
    % 如果是第 3 或 第 4 张图 (底行)，显示 X 轴标签
    if k > 2 
        xlabel('Mass Flow Rate (kg/s)', 'FontSize', 10); 
    end
    
    % 如果是第 1 或 第 3 张图 (左列)，显示 Y 轴标签
    if mod(k, 2) == 1 
        ylabel('Isentropic Efficiency (%)', 'FontSize', 10); 
    end
end

% 显示图例
lgd = legend('show');
lgd.Layout.Tile = 'east';

% 保存图片
exportgraphics(f, out_png, 'Resolution', 300);
fprintf('绘图完成。\n布局: 2x2\n目标变量: 效率 (eta_sX)\n保存为: %s\n', out_png);