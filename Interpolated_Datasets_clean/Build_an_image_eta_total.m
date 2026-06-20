% build_eta_ex_no_filter_simple_axis.m
% 合并多个 CSV，不过滤背压，绘制 eta_ex vs Gc。
% 标题上的背压值自动根据文件名的"第二个数字"确定。
% 横纵坐标使用默认字体，标题保留自定义字号。
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

% 输出文件名设置
out_csv = 'Merged_Results.csv';
out_mat = 'Merged_Results.mat';

%% ========== 初始化结果容器 ==========
Merged = table([],[],[], 'VariableNames', {'source_file','Gc_kg_s','eta_ex_system'});
diagnostics = struct();
detected_back_pressure = 'Unknown'; % 用于存储提取的背压

%% ========== 提取背压用于标题 (取第一个文件的第二个数字) ==========
if ~isempty(csv_files)
    first_fn = csv_files{1};
    % 正则匹配所有连续数字
    nums = regexp(first_fn, '\d+', 'match');
    if length(nums) >= 2
        detected_back_pressure = nums{3}; % 获取第二个数字
    else
        warning('无法从文件名 %s 中提取第二个数字作为背压，将显示 Unknown', first_fn);
    end
end

%% ========== 逐文件处理 ==========
for k = 1:numel(csv_files)
    fn = csv_files{k};
    fprintf('\n=== 处理文件 %d / %d : %s ===\n', k, numel(csv_files), fn);
    diagnostics(k).file = fn;
    
    if ~isfile(fn)
        warning('  文件不存在，跳过：%s', fn);
        diagnostics(k).skipped = true;
        continue;
    end
    
    try
        T = readtable(fn, 'PreserveVariableNames', true);
    catch ME
        warning('  读取 %s 失败：%s. 跳过。', fn, ME.message);
        diagnostics(k).skipped = true;
        continue;
    end
    
    varnames = T.Properties.VariableNames;
    lvar = lower(varnames);
    
    % ---- 找 Gc 列 ----
    idx_gc = find(contains(lvar, 'gc'), 1);
    prefer_gc = {'gc_kg_s','gc_kg','gc'};
    if isempty(idx_gc)
        for p = 1:numel(prefer_gc)
            i = find(strcmpi(varnames, prefer_gc{p}), 1);
            if ~isempty(i), idx_gc = i; break; end
        end
    end
    
    % ---- 找 eta_ex_system 列 ----
    idx_eta = find(contains(lvar, 'eta_ex'), 1);
    if isempty(idx_eta)
        fallbacks = {'eta_system','eta_ex_system','eta_ex'};
        for p = 1:numel(fallbacks)
            i = find(strcmpi(varnames, fallbacks{p}),1);
            if ~isempty(i), idx_eta = i; break; end
        end
    end
    
    if isempty(idx_gc) || isempty(idx_eta)
        warning('  缺少 Gc 或 eta 列，跳过：%s', fn);
        diagnostics(k).skipped = true;
        continue;
    end
    
    % ========== 抽取数值列并转换为 numeric ==========
    % Gc
    Gc_col_raw = T{:, idx_gc};
    try
        if isnumeric(Gc_col_raw)
            Gc_num = double(Gc_col_raw);
        else
            s_gc = string(Gc_col_raw);
            s_gc = strtrim(strrep(s_gc, ',', ''));
            s_gc(s_gc=="") = "NaN";
            Gc_num = double(s_gc);
        end
    catch
        Gc_num = nan(height(T),1);
    end
    
    % eta_ex_system
    eta_col_raw = T{:, idx_eta};
    try
        if isnumeric(eta_col_raw)
            eta_num = double(eta_col_raw);
        else
            s_eta = string(eta_col_raw);
            s_eta = strtrim(strrep(s_eta, ',', ''));
            s_eta(s_eta=="") = "NaN";
            eta_num = double(s_eta);
        end
    catch
        eta_num = nan(height(T),1);
    end
    
    % ========== 仅筛选有效数据 (不再检测背压) ==========
    mask_valid = isfinite(Gc_num) & isfinite(eta_num);
    n_keep = sum(mask_valid);
    
    fprintf('  总行数: %d. 有效数据行数: %d.\n', height(T), n_keep);
    
    if n_keep > 0
        rows_idx = find(mask_valid);
        src_col = repmat(string(fn), numel(rows_idx), 1);
        newT = table(src_col, Gc_num(rows_idx), eta_num(rows_idx), ...
            'VariableNames', {'source_file','Gc_kg_s','eta_ex_system'});
        Merged = [Merged; newT];
    end
    
    diagnostics(k).skipped = false;
    diagnostics(k).n_kept = n_keep;
end % for files

%% ========== 合并后处理与绘图 ==========
if isempty(Merged) || isempty(Merged.Gc_kg_s)
    error('没有有效数据可绘图。请检查 CSV 文件内容。');
end

Merged.source_file = categorical(Merged.source_file);
Merged = sortrows(Merged, {'source_file','Gc_kg_s'});

% 绘图
figure('Units','normalized','Position',[0.15 0.15 0.6 0.6]);
hold on; grid on;
sources = categories(Merged.source_file);
colors = lines(numel(sources));
legends = cell(numel(sources), 1);

for i=1:numel(sources)
    src = sources{i}; 
    idx = Merged.source_file == src;
    X = Merged.Gc_kg_s(idx);
    Y = Merged.eta_ex_system(idx);
    
    % 按 Gc 排序
    [Xs, ord] = sort(X);
    Ys = Y(ord);
    
    plot(Xs, Ys, '-', 'LineWidth', 3, 'MarkerSize', 5, 'Color', colors(i,:));
    
    % Legend 生成逻辑 (取文件名的第一个字符)
    fname_char = char(src); 
    current_legend = ['adjust speeds  ', fname_char(1),'&', fname_char(3)];
    legends{i} = current_legend;
end

% ========== 图表精细化排版 (放大字号，提升学术质感) ==========

% 1. 设置坐标轴名称 (放大至 16 号并加粗)
xlabel('G_c (kg/s)', 'FontSize', 22, 'FontWeight', 'bold');
ylabel('\eta_{ex,system}', 'FontSize', 22, 'FontWeight', 'bold');

% 2. 设置坐标轴刻度数字 (放大至 14 号)
ax = gca; 
ax.FontSize = 21; 
ax.LineWidth = 1.8; % 顺便把坐标轴的边框线稍微加粗，避免线条显单薄

% 3. 设置大标题 (放大至 18 号并加粗)
title_str = sprintf('\\eta_{ex,system} vs G_c (Back Pressure: %s bar)', detected_back_pressure);
title(title_str, 'FontSize', 23, 'FontWeight', 'bold');

% 4. 设置图例 (放大至 12 号)
legend(legends, 'Location', 'best', 'FontSize', 21);

hold off;

%% ========== 保存结果 ==========
try
    Tsave = Merged;
    Tsave.source_file = string(Tsave.source_file);
    writetable(Tsave, out_csv);
    save(out_mat, 'Tsave', 'diagnostics', '-v7.3');
    fprintf('\n合并结果已保存：\n  CSV -> %s\n  MAT -> %s\n', out_csv, out_mat);
catch ME
    warning('保存合并结果失败：%s', ME.message);
end
fprintf('\n完成。\n');