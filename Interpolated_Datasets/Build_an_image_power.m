% build_power_comparison_no_filter.m
% 单张大图：使用超大字号 (22/21/23) 与 2.4 线宽
clearvars; close all; clc;

%% ========== 用户可改项 ==========
csv_files = {
     '1-2-100-compressor.csv'
     '1-3-100-compressor.csv'
     '1-4-100-compressor.csv'
     '2-3-100-compressor.csv'
     '2-4-100-compressor.csv'
     '3-4-100-compressor.csv'
};
out_csv = 'Merged_Power_Results.csv';
out_mat = 'Merged_Power_Results.mat';
Merged = table([],[],[], 'VariableNames', {'source_file','Gc_kg_s','Power_total_W'});
detected_back_pressure = 'Unknown'; 

if ~isempty(csv_files)
    first_fn = csv_files{1};
    nums = regexp(first_fn, '\d+', 'match');
    if length(nums) >= 3, detected_back_pressure = nums{3}; end
end

for k = 1:numel(csv_files)
    fn = csv_files{k};
    if ~isfile(fn), continue; end
    try T = readtable(fn, 'PreserveVariableNames', true); catch, continue; end
    varnames = T.Properties.VariableNames; lvar = lower(varnames);
    
    idx_gc = find(contains(lvar,'gc'),1);
    if isempty(idx_gc), continue; end
    
    power_vals = zeros(height(T), 4);
    for s = 1:4
        idx_p = find(contains(lvar, 'power') & contains(lvar, sprintf('s%d', s)), 1);
        if ~isempty(idx_p), power_vals(:, s) = double(string(T{:, idx_p})); end
    end
    
    Gc = double(string(T{:, idx_gc}));
    Power_total = sum(power_vals, 2, 'omitnan');
    mask = isfinite(Gc) & isfinite(Power_total);
    
    if sum(mask) > 0
        newT = table(repmat(string(fn), sum(mask), 1), Gc(mask), Power_total(mask), ...
            'VariableNames',{'source_file','Gc_kg_s','Power_total_W'} );
        Merged = [Merged; newT];
    end
end

if isempty(Merged), error('没有有效数据'); end
Merged.source_file = categorical(Merged.source_file);
Merged = sortrows(Merged,{'source_file','Gc_kg_s'});

figure('Units','normalized','Position',[0.15 0.15 0.6 0.6]);
hold on; grid on; box on;
sources = categories(Merged.source_file);
colors = lines(numel(sources));
legends = cell(numel(sources), 1);

for i = 1:numel(sources)
    src = sources{i}; idx = Merged.source_file == src;
    X = Merged.Gc_kg_s(idx); Y = Merged.Power_total_W(idx);
    [Xs, ord] = sort(X); Ys = Y(ord);
    
    % 【线宽放大至 2.4】
    plot(Xs, Ys, '-','LineWidth', 3, 'Color', colors(i,:));
    
    fname_char = char(src);
    legends{i} = ['adjust speeds  ', fname_char(1),'&', fname_char(3)];
end

% ========== 图表精细化排版 (单图超大字号) ==========
xlabel('G_c (kg/s)', 'FontSize', 22, 'FontWeight', 'bold');
ylabel('Power_{total} (W)', 'FontSize', 22, 'FontWeight', 'bold');
ax = gca; 
ax.FontSize = 21; 
ax.LineWidth = 1.8; % 边框加粗
title_str = sprintf('Power_{total} vs G_c (Back Pressure: %s bar)', detected_back_pressure);
title(title_str, 'FontSize', 23, 'FontWeight', 'bold');
legend(legends, 'Location', 'best', 'FontSize', 21);
hold off;