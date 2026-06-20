% ================= plot_combined_colors_by_pressure_clean.m =================
% 功能：3x2 布局绘制不同调节方案的转速曲线
% 视觉编码：
%   1. 【颜色】代表【背压】 (100=黑, 90=红, 80=蓝)
%   2. 【过滤】自动隐藏转速恒为 1.0 (不参与调节) 的线条
%   3. 【排版】全局唯一坐标轴标签
% ===================================================================
clearvars; close all; clc;
%% ================= 用户配置区 =================
scheme_prefixes = {'1-2'}; 
target_pressures = {'100', '95', '85'};
file_suffix = '-compressor.csv';
% 颜色配置
pressure_colors = [0 0 0; 1 0 0; 0 0 1]; 
line_style_fixed = '-'; 
out_png = 'combined_colors_clean.png';

%% ================= 绘图逻辑 =================
% 增加 Figure 高度以适应 3 行布局
f = figure('Units', 'normalized', 'Position', [0.05 0.05 0.7 0.9]); 
t = tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% ========== 【排版升级】大标题放大至 23 号 ==========
title(t, 'Active Regulation Speed vs Mass Flow (Fixed Speed Lines Removed)', ...
    'FontSize', 23, 'FontWeight', 'bold');

for k = 1:numel(scheme_prefixes)
    current_scheme = scheme_prefixes{k};
    nexttile; hold on; grid on; box on;
    
    % ========== 【排版升级】子标题放大至 22 号 ==========
    % 将 1-2 格式转换为 1 & 2
    display_scheme = strrep(current_scheme, '-', ' & ');
    title(sprintf('Regulation Stage: %s', display_scheme), 'FontSize', 22, 'FontWeight', 'bold');
    
    for p_idx = 1:numel(target_pressures)
        current_pressure = target_pressures{p_idx};
        curr_color = pressure_colors(p_idx, :); 
        
        fn = sprintf('%s-%s%s', current_scheme, current_pressure, file_suffix);
        if ~isfile(fn), continue; end
        
        opts = detectImportOptions(fn);
        opts.PreserveVariableNames = true;
        T = readtable(fn, opts);
        lvar = lower(T.Properties.VariableNames);
        
        % 找 X 轴
        x_names = {'gc_kg_s', 'gc', 'massflow'};
        idx_x = [];
        for i=1:numel(x_names)
            idx_x = find(strcmpi(lvar, x_names{i}), 1);
            if ~isempty(idx_x), break; end
        end
        if isempty(idx_x), continue; end
        
        % 找 Y 轴
        idx_speeds = zeros(1,4);
        for s=1:4
            patterns = {sprintf('n_s%d',s), sprintf('n_s_%d',s), sprintf('n%d',s)};
            for p=1:numel(patterns)
                idx = find(strcmpi(lvar, patterns{p}), 1);
                if ~isempty(idx), idx_speeds(s) = idx; break; end
            end
        end
        
        X_val = double(T{:, idx_x});
        [X_val, sort_ord] = sort(X_val);
        
        % --- 绘制 S1~S4 ---
        for s = 1:4
            if idx_speeds(s) > 0
                Y_val_raw = double(T{:, idx_speeds(s)});
                Y_val = Y_val_raw(sort_ord);
                mask = isfinite(Y_val);
                
                if ~any(mask), continue; end
                
                % 过滤未调节级 (转速恒定为 1.0)
                if abs(mean(Y_val(mask)) - 1.0) < 1e-3
                    continue; 
                end
                
                % ========== 【排版升级】线宽放大至 3.0 ==========
                plot(X_val(mask), Y_val(mask), ...
                    'Color', curr_color, ...
                    'LineStyle', line_style_fixed, ...
                    'LineWidth', 3, ...
                    'HandleVisibility', 'off'); 
            end
        end
    end
    
    % ========== 【排版升级】设置子图刻度字号与边框线宽 ==========
    ax = gca; 
    ax.FontSize = 21; 
    ax.LineWidth = 1.8;
end

%% ================= 【关键修改】设置全局坐标轴名称 =================
% 作用于布局对象 t，实现全局居中且只出现一次
xlabel(t, 'Mass Flow Rate (kg/s)', 'FontSize', 22, 'FontWeight', 'bold');
ylabel(t, 'Reduced Rotational Speed', 'FontSize', 22, 'FontWeight', 'bold');

%% ================= 生成图例 (位于右侧) =================
hold on; 
legend_handles = [];
legend_labels = {};
for p = 1:numel(target_pressures)
    h = plot(nan, nan, ...
        'Color', pressure_colors(p, :), ...
        'LineWidth', 2.4, ...
        'LineStyle', line_style_fixed); 
    legend_handles(end+1) = h; 
    legend_labels{end+1} = [target_pressures{p} ' bar']; 
end
lgd = legend(legend_handles, legend_labels);
lgd.Layout.Tile = 'east'; 
lgd.FontSize = 21; 

%% ================= 保存 =================
exportgraphics(f, out_png, 'Resolution', 300);
fprintf('绘图完成 (全局坐标轴已设置): %s\n', out_png);