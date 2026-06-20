% ================= plot_four_combinations_eps_3x2.m =================
% 功能：读取 6 个 CSV 文件，绘制 质量流量(X) vs 压比(Y)
% 布局：3行2列
% 对应关系：Stage 1=黑, Stage 2=红, Stage 3=蓝, Stage 4=绿
clearvars; close all; clc;
%% ================= 用户配置区 =================
% 1. 定义文件名列表 (6个文件，对应 3x2 布局)
file_names = {
     '1-2-3-100-compressor.csv'
     '1-2-4-100-compressor.csv'
     '1-3-4-100-compressor.csv'
     '2-3-4-100-compressor.csv'



};
% 2. 定义 X 轴变量名 (Mass Flow Rate)
target_x_candidates = {'gc_kg_s', 'gc', 'massflow', 'mass_flow', 'flow', 'm_dot'}; 
% 3. 输出图片名
out_png = 'six_combinations_eps_pressure_3x2.png';
% 4. 颜色配置 (对应 Stage 1-4)
stage_colors = [0 0 0; 1 0 0; 0 0 1; 0 0.5 0]; 
%% ================= 代码逻辑 =================
f = figure('Units', 'normalized', 'Position', [0.1 0.1 0.7 0.9]); % 稍微加高一点适应3行
t_layout = tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- 处理大标题：提取全局背压 ---
if ~isempty(file_names)
    first_fn = file_names{1};
    global_nums = regexp(first_fn, '\d+', 'match');
    if ~isempty(global_nums)
        global_back_pressure = global_nums{end};
    else
        global_back_pressure = '?';
    end
else
    global_back_pressure = '?';
end

% ========== 【排版升级】大标题放大至 23 号 ==========
main_title_str = sprintf('Compression Ratio vs Mass Flow Rate (Back Pressure: %s bar)', global_back_pressure);
title(t_layout, main_title_str, 'FontSize', 23, 'FontWeight', 'bold');

% --- 循环处理每个文件 ---
for k = 1:numel(file_names)
    raw_fn = file_names{k};
    
    if endsWith(raw_fn, '.csv', 'IgnoreCase', true)
        fn = raw_fn;
        name_for_parsing = raw_fn(1:end-4);
    else
        fn = [raw_fn, '.csv'];
        name_for_parsing = raw_fn;
    end
    
    nums = regexp(name_for_parsing, '\d+', 'match');
    
    if length(nums) >= 2
        stages_str = strjoin(nums(1:end-1), ' & '); 
        subplot_title = sprintf('Regulation: Stages %s', stages_str);
    else
        subplot_title = sprintf('Case %d', k);
    end
    
    nexttile; hold on; grid on; box on;
    % ========== 【排版升级】子标题放大至 22 号 ==========
    title(subplot_title, 'FontSize', 21, 'FontWeight', 'bold');
    
    if ~isfile(fn)
        text(0.5, 0.5, 'File Not Found', 'HorizontalAlignment', 'center', 'Color', 'r');
        warning('文件不存在: %s', fn);
        continue;
    end
    
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
    
    % --- 匹配 Y 轴 ---
    eps_idxs = zeros(1,4);
    for s = 1:4
        patterns = {sprintf('eps_s%d',s), sprintf('esp_s%d',s), sprintf('eps%d',s), sprintf('ratio%d',s)};
        for p = 1:numel(patterns)
            idx = find(strcmpi(lvar, patterns{p}), 1); 
            if isempty(idx), idx = find(contains(lvar, patterns{p}), 1); end 
            if ~isempty(idx), eps_idxs(s) = idx; break; end
        end
    end
    
    if isempty(x_idx) || all(eps_idxs == 0)
        continue;
    end
    
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
                 'LineWidth', 3, 'DisplayName', sprintf('Stage %d', level));
        end
    end
    
    % ========== 【排版升级】子图坐标轴线宽与刻度适配 ==========
    % 注意：这里删除了原来单独设置 xlabel 和 ylabel 的 if 语句
    ax = gca; 
    ax.FontSize = 21; 
    ax.LineWidth = 1.8;
end

% ================= 【排版升级】设置全局唯一的坐标轴名称 =================
% 通过将标签赋给 t_layout，MATLAB 会自动在整个大图的底部和左侧居中显示
xlabel(t_layout, 'Mass Flow Rate (kg/s)', 'FontSize', 21, 'FontWeight', 'bold');
ylabel(t_layout, 'Pressure Ratio (\epsilon)', 'FontSize', 21, 'FontWeight', 'bold');

% 显示图例
lgd = legend('show');
lgd.Layout.Tile = 'east';
% ========== 【排版升级】图例字号放大 ==========
lgd.FontSize = 21; 

% 保存图片
exportgraphics(f, out_png, 'Resolution', 300);
fprintf('绘图完成。\n布局: 3x2 全局坐标轴\n保存为: %s\n', out_png);