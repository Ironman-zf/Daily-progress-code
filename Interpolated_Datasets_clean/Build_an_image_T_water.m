% build_Tmix_vs_Gc_3_pressures_centered.m
% 多子图布局：使用等效缩放的粗体和加粗线条，防止文字重叠
clearvars; close all; clc;
target_pressures = {'100', '90', '80'}; 
scheme_prefixes = {'1-2', '1-3', '1-4', '2-3','2-4','3-4'}; % 若要画1-2-3，改这里即可
file_suffix = '-compressor.csv';
scheme_colors = lines(numel(scheme_prefixes)); 
out_png = 'Tmix_vs_Gc_Centered_NoStretch.png';
f = figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]); 
t = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
% 【大标题字号放大】
title(t, 'Water Mixing Temperature vs Mass Flow Rate', 'FontSize', 22, 'FontWeight', 'bold');
for p_idx = 1:numel(target_pressures)
    current_P = target_pressures{p_idx};
    if p_idx == 1, nexttile(1, [1, 2]); 
    elseif p_idx == 2, nexttile(3, [1, 2]); 
    elseif p_idx == 3, nexttile(6, [1, 2]); end
    
    hold on; grid on; box on;
    lines_handles = gobjects(0); legend_str = {};
    
    for s_idx = 1:numel(scheme_prefixes)
        current_scheme = scheme_prefixes{s_idx};
        fn = sprintf('%s-%s%s', current_scheme, current_P, file_suffix);
        if ~isfile(fn), continue; end
        
        try
            T = readtable(fn, 'PreserveVariableNames', true);
            lvar = lower(T.Properties.VariableNames);
            
            % 1. 寻找流量列
            idx_gc = find(contains(lvar, 'gc_kg_s') | contains(lvar, 'gc'), 1);
            if isempty(idx_gc), continue; end
            Gc = double(string(T{:, idx_gc}));
            
            % ---------------------------------------------------------
            % 2. 核心反算逻辑：动态提取各级出水温度并求均值
            % ---------------------------------------------------------
            % 匹配所有形如 T_water_out_s1_K 的列
            idx_stages_water = find(contains(lvar, 't_water_out_s') & contains(lvar, '_k'));
            
            if ~isempty(idx_stages_water)
                % 提取所有级数的水温矩阵 (N行 x 级数列)
                T_water_stages = double(string(T{:, idx_stages_water}));
                % 按行求平均（基于等分流量假设），'omitnan' 确保即使某级崩溃为 NaN，也不会污染最终的混合温度
                Tmix = mean(T_water_stages, 2, 'omitnan');
            else
                % 兜底保护：如果文件里连分级出水温度都没有，尝试读原有的 t_mix
                idx_tmix = find(contains(lvar, 't_mix') | contains(lvar, 't_water_mix'), 1);
                if isempty(idx_tmix), continue; end
                Tmix = double(string(T{:, idx_tmix}));
            end
            % ---------------------------------------------------------

            mask = isfinite(Gc) & isfinite(Tmix);
            [Gc_s, ord] = sort(Gc(mask)); Tmix_s = Tmix(ord);
            
            % 【子图线宽放大至 2.0】
            h = plot(Gc_s, Tmix_s, '-', 'Color', scheme_colors(s_idx, :), 'LineWidth', 2.4);
            lines_handles(end+1) = h(1); %#ok<*AGROW>
            legend_str{end+1} = ['Scheme ', current_scheme];
        catch, end
    end
    
    % ========== 子图精细化排版 ==========
    xlabel('G_c (kg/s)', 'FontSize', 22, 'FontWeight', 'bold');
    ylabel('T_{mix} (K)', 'FontSize', 22, 'FontWeight', 'bold');
    title(sprintf('Back Pressure: %s bar', current_P), 'FontSize', 23, 'FontWeight', 'bold');
    ax = gca; ax.FontSize = 21; ax.LineWidth = 1.8; % 刻度放大，边框加粗
    
    if ~isempty(lines_handles)
        legend(lines_handles, legend_str, 'Location', 'best', 'FontSize', 21);
    end
end
exportgraphics(f, out_png, 'Resolution', 300);