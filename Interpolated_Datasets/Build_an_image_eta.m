% ================= plot_six_combinations_eff_3x2.m =================
% 功能：读取 6 个 CSV 文件，绘制 质量流量(X) vs 等熵效率(Y)
% 布局：3行2列，全局唯一坐标轴标签，自动提取文件名中的背压
% 对应关系：Stage 1=黑, Stage 2=红, Stage 3=蓝, Stage 4=绿
clearvars; close all; clc;

%% ================= 用户配置区 =================
file_names = {     
     '1-2-80robust_results.csv'
     '1-3-80robust_results.csv'
     '1-4-80robust_results.csv'
     '2-3-80robust_results.csv'
     '2-4-80robust_results.csv'
     '3-4-80robust_results.csv'
};
target_x_candidates = {'gc_kg_s', 'gc'}; 
stage_colors = [0 0 0; 1 0 0; 0 0 1; 0 0.5 0]; 

f = figure('Units', 'normalized', 'Position', [0.1 0.05 0.7 1]); 
% 使用 tiledlayout 布局
t_layout = tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% ========== 【智能升级】自动从第一个文件名中提取背压名称 ==========
if ~isempty(file_names)
    % 提取文件名中所有的数字序列
    nums_extracted = regexp(file_names{1}, '\d+', 'match');
    if ~isempty(nums_extracted)
        % 按照命名习惯，最后一个数字通常是背压 (例如 1-2-90 中的 90)
        global_back_pressure = nums_extracted{end}; 
    else
        global_back_pressure = 'Unknown';
    end
else
    global_back_pressure = 'None';
end

% ========== 【排版升级】大标题放大至 23 号 ==========
title_str = sprintf('Compressor Efficiency vs Mass Flow Rate (Back Pressure: %s bar)', global_back_pressure);
title(t_layout, title_str, 'FontSize', 23, 'FontWeight', 'bold');

%% ================= 核心绘图逻辑 =================
for k = 1:numel(file_names)
    fn = file_names{k}; 
    name_for_parsing = strrep(fn, '.csv', '');
    % 提取子图标题所需的级数数字
    nums = regexp(name_for_parsing, '\d+', 'match');
    
    if length(nums) >= 2
        % 取除最后一个（背压）以外的所有数字作为调节级
        subplot_title = sprintf('Regulation: Stages %s', strjoin(nums(1:end-1), ' & '));
    else
        subplot_title = sprintf('Case %d', k);
    end
    
    nexttile; hold on; grid on; box on;
    % ========== 【排版升级】子标题放大至 21 号 ==========
    title(subplot_title, 'FontSize', 21, 'FontWeight', 'bold');
    
    if ~isfile(fn)
        text(0.5, 0.5, 'File Not Found', 'HorizontalAlignment', 'center', 'FontSize', 14);
        continue; 
    end
    
    opts = detectImportOptions(fn); 
    opts.PreserveVariableNames = true; 
    T = readtable(fn, opts); 
    lvar = lower(T.Properties.VariableNames); 
    
    % 匹配流量 X 轴
    x_idx = find(contains(lvar, 'gc'), 1);
    if isempty(x_idx), continue; end
    
    X_Data = double(string(T{:, x_idx}));
    [X_Data, sort_idx] = sort(X_Data);
    
    % 绘制四个阶段的效率
    for s = 1:4
        eta_idx = find(contains(lvar, sprintf('eta_s%d',s)) | ...
                       contains(lvar, sprintf('eta%d',s)) | ...
                       contains(lvar, sprintf('eff%d',s)), 1);
        if ~isempty(eta_idx)
            Eta_val = double(string(T{:, eta_idx})); 
            Eta_val = Eta_val(sort_idx);
            mask = isfinite(X_Data) & isfinite(Eta_val);
            if any(mask)
                % 【线宽放大至 3，保持粗壮感】
                plot(X_Data(mask), Eta_val(mask), 'Color', stage_colors(s, :), ...
                     'LineWidth', 3, 'DisplayName', sprintf('Stage %d', s));
            end
        end
    end
    
    % ========== 【排版升级】设置子图刻度字号与线宽 ==========
    ax = gca; 
    ax.FontSize = 21; 
    ax.LineWidth = 1.8; 
end

%% ================= 【全局排版升级】设置全局坐标轴与图例 =================
% 1. 设置全局唯一的 X 和 Y 轴标签 (21 号加粗)
xlabel(t_layout, 'Mass Flow Rate (kg/s)', 'FontSize', 21, 'FontWeight', 'bold');
ylabel(t_layout, 'Isentropic Efficiency (%)', 'FontSize', 21, 'FontWeight', 'bold');

% 2. 统一图例设置 (21 号)
lgd = legend('show'); 
lgd.Layout.Tile = 'east'; 
lgd.FontSize = 21; 

% 保存结果
out_png = 'six_combinations_efficiency_3x2.png';
exportgraphics(f, out_png, 'Resolution', 300);
fprintf('绘图完成。检测到背压: %s bar。保存为: %s\n', global_back_pressure, out_png);