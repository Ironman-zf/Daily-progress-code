% build_power_vs_exergy_efficiency.m
% 单张大图：展示 总功率 与 㶲效率 的关系
% 视觉标准：使用超大字号 (22/21/23) 与 3.0 线宽
clearvars; close all; clc;

%% ========== 1. 用户配置区 ==========
csv_files = {
     '1-2-100-compressor.csv'
     '1-3-100-compressor.csv'
     '1-4-100-compressor.csv'
     '2-3-100-compressor.csv'
     '2-4-100-compressor.csv'
     '3-4-100-compressor.csv'




};

% 初始化一个表来存储合并后的数据，包含文件源、总功率和㶲效率
Merged = table([],[],[], 'VariableNames', {'source_file', 'Power_total_W', 'eta_ex_system'});

% 尝试从第一个文件名中提取目标压力作为标题信息
detected_pressure = 'Unknown'; 
if ~isempty(csv_files)
    first_fn = csv_files{1};
    nums = regexp(first_fn, '\d+', 'match');
    % 文件名如 4-100-compressor.csv，提取第 2 个数字作为压力
    if length(nums) >= 2
        detected_pressure = nums{3}; 
    end
end

%% ========== 2. 自动化数据读取与清洗 ==========
for k = 1:numel(csv_files)
    fn = csv_files{k};
    if ~isfile(fn)
        fprintf('警告：文件不存在 %s\n', fn);
        continue; 
    end
    
    try 
        T = readtable(fn, 'PreserveVariableNames', true); 
    catch
        fprintf('警告：读取失败 %s\n', fn);
        continue; 
    end
    
    varnames = T.Properties.VariableNames; 
    lvar = lower(varnames);
    
    % 寻找代表总功率的列 (Power_total_W)
    idx_power = find(strcmpi(lvar, 'power_total_w'), 1);
    % 寻找代表㶲效率的列 (eta_ex_system)
    idx_eta = find(strcmpi(lvar, 'eta_ex_system'), 1);
    
    if isempty(idx_power) || isempty(idx_eta)
        fprintf('警告：%s 中未找到所需的 Power_total_W 或 eta_ex_system 列\n', fn);
        continue; 
    end
    
    % 提取数据并转为双精度数值
    Power_val = double(string(T{:, idx_power}));
    Eta_val = double(string(T{:, idx_eta}));
    
    % 过滤掉 NaN 或 Inf 等非法数据点
    mask = isfinite(Power_val) & isfinite(Eta_val);
    
    if sum(mask) > 0
        % 针对同一个文件的数据建立新表
        newT = table(repmat(string(fn), sum(mask), 1), Power_val(mask), Eta_val(mask), ...
            'VariableNames', {'source_file', 'Power_total_W', 'eta_ex_system'});
        Merged = [Merged; newT];
    end
end

if isempty(Merged)
    error('所有文件均未能提取出有效的 Power_total_W 和 eta_ex_system 数据。'); 
end

%% ========== 3. 数据排序与绘图初始化 ==========
Merged.source_file = categorical(Merged.source_file);

% 创建图形窗口，设置合适的比例
figure('Units','normalized','Position',[0.15 0.15 0.6 0.6]);
hold on; grid on; box on;

sources = categories(Merged.source_file);
% 使用 MATLAB 推荐的科学对比色卡
colors = lines(numel(sources));
legends = cell(numel(sources), 1);

%% ========== 4. 循环绘制每组数据的曲线 ==========
for i = 1:numel(sources)
    src = sources{i}; 
    idx = Merged.source_file == src;
    
    % 提取当前文件的 X (功率) 和 Y (㶲效率)
    X = Merged.Power_total_W(idx); 
    Y = Merged.eta_ex_system(idx);
    
    % 必须将 X 轴（功率）按从小到大排序，否则画出的线会乱飞
    [Xs, ord] = sort(X); 
    Ys = Y(ord);
    
    % 绘制线条，线宽按照之前设置的 3.0
    plot(Xs, Ys, '-', 'LineWidth', 3, 'Color', colors(i,:));
    
    % --- 动态生成图例 ---
    % 从文件名如 '4-100-compressor.csv' 中提取方案号
    fname_char = char(src);
    file_nums = regexp(fname_char, '\d+', 'match');
    if ~isempty(file_nums)
         % 提取方案编号，例如 '4' 或 '3'
        legends{i} = ['Adjust VSV ', file_nums{1} '&' file_nums{2}];
    else
        legends{i} = fname_char;
    end
end

%% ========== 5. 图表精细化排版 ==========
% 设置 X 轴标签：总功率 (包含支持 LaTeX 的下划线)
xlabel('Total Power (W)', 'FontSize', 22, 'FontWeight', 'bold');

% 设置 Y 轴标签：㶲效率
ylabel('Exergy Efficiency \eta_{ex}', 'FontSize', 22, 'FontWeight', 'bold');

% 坐标轴刻度字体及边框加粗
ax = gca; 
ax.FontSize = 21; 
ax.LineWidth = 1.8; 

% 给 Y 轴预留一点呼吸空间，防止曲线贴顶
y_limits = ylim;
ylim([y_limits(1), y_limits(2) + (y_limits(2)-y_limits(1))*0.15]);

% 动态主标题
title_str = sprintf('Exergy Efficiency vs Total Power (Pressure: %s bar)', detected_pressure);
title(title_str, 'FontSize', 23, 'FontWeight', 'bold');

% 图例设置
legend(legends, 'Location', 'best', 'FontSize', 21);
hold off;

fprintf('绘图完成。已应用 X 轴：功率，Y 轴：㶲效率。\n');