% ================= plot_four_combinations_eps_2x2.m =================
% 功能：读取 4 个 CSV 文件，绘制 质量流量(X) vs 压比(Y)
% 布局：2行2列 (上二下二)，每个文件一张子图
% 对应关系：Stage 1=黑, Stage 2=红, Stage 3=蓝, Stage 4=绿
clearvars; close all; clc;

%% ================= 用户配置区 =================
% 1. 定义文件名列表 (4个文件，对应 2x2 布局)
file_names = {
     '1-3-4-95-compressor.csv'
     '1-3-4-85-compressor.csv'
     '1-3-4-103-compressor.csv'






 

};

% 2. 定义 X 轴变量名 (Mass Flow Rate)
target_x_candidates = {'gc_kg_s', 'gc', 'massflow', 'mass_flow', 'flow', 'm_dot'}; 

% 3. 输出图片名
out_png = 'four_combinations_eps_pressure_2x2.png';

% 4. 颜色配置 (对应 Stage 1-4)
stage_colors = [0 0 0; 1 0 0; 0 0 1; 0 0.5 0]; 

%% ================= 代码逻辑 =================
f = figure('Units', 'normalized', 'Position', [0.1 0.1 0.7 0.8]); % 稍微调高一点比例适应2x2
% 【修改点1】布局改为 2行 2列
t_layout = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- 处理大标题：提取全局背压 ---
if ~isempty(file_names)
    first_fn = file_names{1};
    global_nums = regexp(first_fn, '\d+', 'match');
    % 尝试智能获取背压：取最后一个数字作为背压 (针对 1-2-3-100 这种情况)
    if ~isempty(global_nums)
        global_back_pressure = global_nums{end-1};
    else
        global_back_pressure = '?';
    end
else
    global_back_pressure = '?';
end

% 设置大标题
main_title_str = sprintf('Expansion Ratio vs Mass Flow Rate (Inlet pressure: %s bar)', global_back_pressure);
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
    
    % --- 提取数字作为子标题 (支持任意数量的调速级) ---
    nums = regexp(name_for_parsing, '\d+', 'match');
    
    % 假设最后一个数字是压力(100)，前面的都是级数
    if length(nums) >= 2
        % 将除最后一个数字外的所有数字连接起来 (e.g., "1", "2", "3")
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
    
    % --- 匹配 X 轴 (流量) ---
    x_idx = [];
    for i = 1:length(target_x_candidates)
        candidate = lower(target_x_candidates{i});
        idx = find(strcmpi(lvar, candidate), 1);
        if isempty(idx), idx = find(contains(lvar, candidate), 1); end
        if ~isempty(idx), x_idx = idx; break; end
    end
    
    % --- 匹配 Y 轴 (压比: eps_s1, eps_s2...) ---
    eps_idxs = zeros(1,4);
    for s = 1:4
        patterns = {
            sprintf('eps_s%d',s), ...
            sprintf('PR_s%d',s), ...   
            sprintf('eps%d',s), ...     
            sprintf('ratio%d',s)
        };
        for p = 1:numel(patterns)
            idx = find(strcmpi(lvar, patterns{p}), 1); 
            if isempty(idx), idx = find(contains(lvar, patterns{p}), 1); end 
            if ~isempty(idx), eps_idxs(s) = idx; break; end
        end
    end
    
    if isempty(x_idx) || all(eps_idxs == 0)
        text(0.5, 0.5, 'Data Missing', 'HorizontalAlignment', 'center', 'Color', 'r');
        continue;
    end
    
    % --- 提取并排序 ---
    X_Data = double(T{:, x_idx});
    
    EpsData = nan(height(T), 4);
    for s = 1:4
        if eps_idxs(s) > 0
            EpsData(:, s) = double(T{:, eps_idxs(s)});
        end
    end
    
    [X_Data, sort_idx] = sort(X_Data);
    EpsData = EpsData(sort_idx, :);
    
    % --- 绘制曲线 ---
    for level = 1:4
        if ~isnan(EpsData(1, level))
            plot(X_Data, EpsData(:, level), 'Color', stage_colors(level, :), ...
                 'LineWidth', 1.5, 'DisplayName', sprintf('Stage %d', level));
        end
    end
    
    % --- 【修改点2】坐标轴标签逻辑适应 2x2 ---
    % 如果是第 3 或 第 4 张图 (底行)，显示 X 轴标签
    if k > 2 
        xlabel('Mass Flow Rate (kg/s)', 'FontSize', 10); 
    end
    
    % 如果是第 1 或 第 3 张图 (左列)，显示 Y 轴标签 (mod(1,2)=1, mod(3,2)=1)
    if mod(k, 2) == 1 
        ylabel('Pressure Ratio (\pi)', 'FontSize', 10); 
    end
end

% 显示图例
lgd = legend('show');
lgd.Layout.Tile = 'east';

% 保存图片
exportgraphics(f, out_png, 'Resolution', 300);
fprintf('绘图完成。\n布局: 2x2\n保存为: %s\n', out_png);