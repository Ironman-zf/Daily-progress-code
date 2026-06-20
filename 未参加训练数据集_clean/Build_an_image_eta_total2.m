% ================= build_eta_ex_no_filter_simple_axis.m =================
% 功能：合并多个 CSV，绘制指定效率列 vs Gc 的关系。
% 视觉标准：标题 23pt, 标签 22pt, 刻度/图例 21pt, 线宽 3.0, 边框 1.8
% 科学优化：目标变量全局定义，解决变量名冲突，修正正则提取逻辑
% ========================================================================
clearvars; close all; clc;

%% ========== 1. 用户配置区 (变量自解释) ==========
csv_files = {
     '1-2-95-compressor.csv'
     '1-2-85-compressor.csv'



};

out_csv = 'Merged_Results.csv';
out_mat = 'Merged_Results.mat';

% 核心优化：将自变量和因变量均设为全局变量，彻底杜绝硬编码引发的崩溃
target_gc_col = 'Gc_kg_s'; 
target_eta_col = 'eta_ex_system'; 
target_eta_label = '\eta_{ex,system}'; 

%% ========== 2. 初始化与智能参数提取 ==========
Merged = table([], [], [], 'VariableNames', {'source_file', target_gc_col, target_eta_col});
diagnostics = struct();
detected_back_pressure = 'Unknown';

% 提取压力用于标题：在 '1-2-70' 中，70 是第 3 个数字
if ~isempty(csv_files)
    first_fn = csv_files{1};
    nums_global = regexp(first_fn, '\d+', 'match');
    if length(nums_global) >= 3
        detected_back_pressure = nums_global{3}; 
    else
        warning('无法从文件名中提取压力参数，将显示 Unknown');
    end
end

%% ========== 3. 逐文件数据清洗与提取 ==========
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
    
    % 寻找 Gc 列
    idx_gc = find(strcmpi(varnames, target_gc_col), 1);
    if isempty(idx_gc)
        idx_gc = find(contains(lvar, 'gc'), 1); 
    end
    
    % 寻找目标 eta 列
    idx_eta = find(strcmpi(varnames, target_eta_col), 1);
    
    if isempty(idx_gc) || isempty(idx_eta)
        warning('  未找到必要的列，跳过以保证数据准确性。');
        diagnostics(k).skipped = true;
        continue;
    end
    
    % 抽取并转换数值（利用向量化函数替代冗长的 try-catch）
    Gc_col_raw = T{:, idx_gc};
    if isnumeric(Gc_col_raw)
        Gc_num = double(Gc_col_raw);
    else
        Gc_num = str2double(strrep(string(Gc_col_raw), ',', ''));
    end
    
    eta_col_raw = T{:, idx_eta};
    if isnumeric(eta_col_raw)
        eta_num = double(eta_col_raw);
    else
        eta_num = str2double(strrep(string(eta_col_raw), ',', ''));
    end
    
    % 数据清洗与筛选
    mask_valid = isfinite(Gc_num) & isfinite(eta_num);
    n_keep = sum(mask_valid);
    
    fprintf('  总行数: %d. 有效数据行数: %d.\n', height(T), n_keep);
    
    if n_keep > 0
        rows_idx = find(mask_valid);
        src_col = repmat(string(fn), numel(rows_idx), 1);
        
        newT = table(src_col, Gc_num(rows_idx), eta_num(rows_idx), ...
            'VariableNames', {'source_file', target_gc_col, target_eta_col});
        Merged = [Merged; newT];
    end
    
    diagnostics(k).skipped = false;
    diagnostics(k).n_kept = n_keep;
end 

%% ========== 4. 绘图与排版逻辑 ==========
% 此处完全使用动态变量判断，不写死任何列名
if isempty(Merged) || isempty(Merged.(target_gc_col))
    error('没有有效数据可绘图。请检查 CSV 文件内容。');
end

Merged.source_file = categorical(Merged.source_file);
Merged = sortrows(Merged, {'source_file', target_gc_col});

figure('Units','normalized','Position',[0.15 0.15 0.5 0.6]);
hold on; grid on; box on;

sources = categories(Merged.source_file);
colors = lines(numel(sources));
legends = cell(numel(sources), 1);

for i = 1:numel(sources)
    src = sources{i}; 
    idx = Merged.source_file == src;
    
    % 动态获取行列
    X = Merged.(target_gc_col)(idx);
    Y = Merged.(target_eta_col)(idx); 
    
    [Xs, ord] = sort(X);
    Ys = Y(ord);
    
    plot(Xs, Ys, '-', 'LineWidth', 3, 'Color', colors(i,:));
    
    % 修正图例逻辑：只提取前两个数字表征调节级数
    nums_legend = regexp(char(src), '\d+', 'match');
    if length(nums_legend) >= 2
        legends{i} = sprintf('adjust VSV %s&%s', nums_legend{1}, nums_legend{2},nums_legend{3});
    else
        legends{i} = char(src);
    end
end

% ========== 图表学术规范设置 ==========
xlabel('G_c (kg/s)', 'FontSize', 22, 'FontWeight', 'bold');
ylabel(target_eta_label, 'FontSize', 22, 'FontWeight', 'bold');

ax = gca; 
ax.FontSize = 21; 
ax.LineWidth = 1.8; 

title_str = sprintf('%s vs G_c (Inlet pressure: %s bar)', target_eta_label, detected_back_pressure);
title(title_str, 'FontSize', 23, 'FontWeight', 'bold');

legend(legends, 'Location', 'best', 'FontSize', 21);
hold off;

%% ========== 5. 保存结果 ==========
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