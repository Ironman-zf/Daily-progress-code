% build_eta_ex_no_filter_simple_axis.m
% 合并多个 CSV，绘制指定效率列 vs Gc 的关系。
clearvars; close all; clc;

%% ========== 用户配置区 (变量自解释) ==========
csv_files = {
     '1-2-3-90-_111Results_Repaired.csv'
     '1-2-4-90-_111Results_Repaired.csv'
     '1-3-4-90-_111Results_Repaired.csv'
     '2-3-4-90-_111Results_Repaired.csv'
};

% 输出文件名设置
out_csv = 'Merged_Results.csv';
out_mat = 'Merged_Results.mat';

% 核心优化：将目标效率列提取为全局变量，避免硬编码导致的前后不一致
target_eta_col = 'eta_ex_system_1'; 
target_eta_label = '\eta_{ex,system 1}'; % 用于绘图的标签显示

%% ========== 初始化结果容器 ==========
% 表头直接引用配置变量，保证一致性
Merged = table([], [], [], 'VariableNames', {'source_file', 'Gc_kg_s', target_eta_col});
diagnostics = struct();
detected_back_pressure = 'Unknown';

%% ========== 提取背压用于标题 ==========
if ~isempty(csv_files)
    first_fn = csv_files{1};
    nums = regexp(first_fn, '\d+', 'match');
    if length(nums) >= 2
        detected_back_pressure = nums{4}; 
    else
        warning('无法从文件名 %s 中提取背压参数，将显示 Unknown', first_fn);
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
        warning('  读取失败：%s. 跳过。', ME.message);
        diagnostics(k).skipped = true;
        continue;
    end
    
    varnames = T.Properties.VariableNames;
    lvar = lower(varnames);
    
    % ---- 1. 寻找 Gc 列 (精确匹配优先) ----
    idx_gc = find(strcmpi(varnames, 'gc_kg_s'), 1);
    if isempty(idx_gc)
        idx_gc = find(contains(lvar, 'gc'), 1); % 备用模糊匹配
    end
    
    % ---- 2. 寻找目标 eta 列 (科学严谨：必须精确匹配目标变量) ----
    idx_eta = find(strcmpi(varnames, target_eta_col), 1);
    if isempty(idx_eta)
        warning('  未找到精确列名 %s，跳过该文件以保证数据准确性。', target_eta_col);
        diagnostics(k).skipped = true;
        continue;
    end
    
    if isempty(idx_gc)
        warning('  缺少 Gc 列，跳过：%s', fn);
        diagnostics(k).skipped = true;
        continue;
    end
    
    % ========== 抽取数值列并转换为 numeric (增强鲁棒性) ==========
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
    
    % ========== 数据清洗与筛选 ==========
    mask_valid = isfinite(Gc_num) & isfinite(eta_num);
    n_keep = sum(mask_valid);
    
    fprintf('  总行数: %d. 有效数据行数: %d.\n', height(T), n_keep);
    
    if n_keep > 0
        rows_idx = find(mask_valid);
        src_col = repmat(string(fn), numel(rows_idx), 1);
        
        % 此处建表也直接引用目标列名，绝不手写字符串
        newT = table(src_col, Gc_num(rows_idx), eta_num(rows_idx), ...
            'VariableNames', {'source_file', 'Gc_kg_s', target_eta_col});
        Merged = [Merged; newT];
    end
    
    diagnostics(k).skipped = false;
    diagnostics(k).n_kept = n_keep;
end 

%% ========== 合并后处理与绘图 ==========
if isempty(Merged) || isempty(Merged.Gc_kg_s)
    error('没有有效数据可绘图。请检查 CSV 文件内容。');
end

Merged.source_file = categorical(Merged.source_file);
Merged = sortrows(Merged, {'source_file', 'Gc_kg_s'});

figure('Units','normalized','Position',[0.15 0.15 0.6 0.6]);
hold on; grid on;

sources = categories(Merged.source_file);
colors = lines(numel(sources));
legends = cell(numel(sources), 1);

for i = 1:numel(sources)
    src = sources{i}; 
    idx = Merged.source_file == src;
    
    X = Merged.Gc_kg_s(idx);
    % 动态引用目标列，取代之前的 Merged.eta_ex_system2
    Y = Merged.(target_eta_col)(idx); 
    
    [Xs, ord] = sort(X);
    Ys = Y(ord);
    
    plot(Xs, Ys, '-', 'LineWidth', 1.4, 'MarkerSize', 5, 'Color', colors(i,:));
    
    fname_char = char(src); 
    legends{i} = ['adjust the VSV of ', fname_char(1),'&', fname_char(3),'&', fname_char(5)];
end

xlabel('G_c (kg/s)');
ylabel(target_eta_label);

title_str = sprintf('%s vs G_c (Inlet pressure: %s bar)', target_eta_label, detected_back_pressure);
title(title_str, 'FontSize', 14);
legend(legends, 'Location', 'best');
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