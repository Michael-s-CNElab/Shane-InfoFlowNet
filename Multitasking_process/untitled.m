cb = colorbar; % 获取 colorbar 对象
limits = cb.Limits;
colormap_data = colormap;

% GC Pz -> Fz 71 80 242  0.2773, 0.3125, 0.9453
% GC Cz -> Fz 71 78 241  0.2773, 0.3047, 0.9414

% TCDF Cz -> Fz 57 200 148 0.2227 0.7813 0.5781

%% Multitasking MVGC result
windows = 15;
subject = 36;
channel = 4;
trial = 30;
order = 29;
ch_location = {'Fz', 'Cz', 'Pz', 'Oz'};

MT_MVGC_result = zeros(channel, channel, windows, trial, subject);

mat_path = 'I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\data\Multitasking\csvdata\';
mat_files = dir([mat_path, '*.csv']);

for i=1:length(mat_files)
    name_spilt = split(mat_files(i).name, ["_", "."]);
    sub = str2num(name_spilt{1});
    tr = str2num(name_spilt{2});
    data = readmatrix(strcat(mat_path, mat_files(i).name));
    for w=1:windows
        win_data = data((w*100)-99:w*100, :);
        MT_MVGC_result(:, :, w, tr, sub) = MVGC_timedomain_GC(win_data', 1, order);
    end
end

colmin = min(MT_MVGC_result);
colmax = max(MT_MVGC_result);
MT_MVGC_result_scale = rescale(MT_MVGC_result,"InputMin",colmin,"InputMax",colmax);
mean_GC_result = mean(mean(MT_MVGC_result_scale, 4), 5);

err_GC_result = mean_GC_result(:, :, 7) - mean_GC_result(:, :, 1);

H_array = zeros(4, 4);

for c_0=1:channel
    for c=1:channel
        RT_w = squeeze(mean(MT_MVGC_result(c_0, c, 7, :, :), 4));
        BSL_w = squeeze(mean(MT_MVGC_result(c_0, c, 1, :, :), 4));
        H_array(c_0, c) = ttest(RT_w, BSL_w);
    end
end

mask = H_array == 0;

weight = err_GC_result;
weight(mask) = 0;
weight = weight';
MBC = MyBiChordChart(weight, ch_location);
set(gcf,'position',[100,100,500, 500], 'Color', 'none');
clim([-1 1])


%% Multitasking TCDF result
csvPath = 'J:\TCDF_result\';
TCDF_result = dir([csvPath, '*_Attention_Scores.csv']);

windows = 15;
subject = 36;
channel = 4;
trial = 30;
ch_location = {'Fz', 'Cz', 'Pz', 'Oz'};

MT_TCDF_result = zeros(channel, channel, windows, trial*subject);

for i=1:length(TCDF_result)
    name_spilt = split(TCDF_result(i).name, ["_", "."]);
    sub = str2num(name_spilt{1});
    w = str2num(name_spilt{2}(1:2))+1;
    MT_TCDF_result(:, :, w, sub) = readmatrix(strcat(csvPath, TCDF_result(i).name));
    disp([TCDF_result(i).name, '--', num2str(w)]);
end

MT_TCDF_result = reshape(MT_TCDF_result, [channel, channel, windows, trial, subject]);

for sub=1:subject
    for tr=1:trial
        for w=1:windows
            score = MT_TCDF_result(:, :, w, tr, sub);

            min_score = min(score, [], "all");
            max_score = max(score, [], "all");

            MT_TCDF_result(:, :, w, tr, sub) = (score - min_score) / (max_score - min_score);

            disp([num2str(sub), '--', num2str(tr), '--', num2str(w)]);
        end
    end
end

mean_TCDF_result = mean(mean(MT_TCDF_result, 4), 5);
err_TCDF_result = mean_TCDF_result(:, :, 7) - mean_TCDF_result(:, :, 1);

H_array = zeros(4, 4);

for c_0=1:channel
    for c=1:channel
        RT_w = squeeze(mean(MT_TCDF_result(c_0, c, 7, :, :), 4));
        BSL_w = squeeze(mean(MT_TCDF_result(c_0, c, 1, :, :), 4));
        H_array(c_0, c) = ttest(RT_w, BSL_w);
    end
end

mask = H_array == 0;

weight = err_TCDF_result;
weight(mask) = 0;
weight = weight';
MBC = MyBiChordChart(weight, ch_location);
set(gcf,'position',[100,100,500, 500], 'Color', 'none');
clim([-1 1])
%% Multitasking signal compare
%% no attn plot
x = 1:1:100;
t = tiledlayout(2, 1);

origin_color = [0.167 0.794 0.310];
shuffle_color = [0.067 0.294 0.610];
predict_color = [0.767 0.294 0.410];
mask_color = [0.7 0.9 0];

nexttile;
plot(x, shuffle_signal(1, :), "Color", shuffle_color, 'LineWidth', 2, 'LineStyle', '--');
hold on;
plot(x, signal(1, :), "Color", origin_color, 'LineWidth', 2, 'LineStyle', '--');
hold on;
plot(x, Noattn_predict(1, :), "Color", predict_color, 'LineWidth', 2);
ylim([-3 3]);
yticks([-2.0 0. 2.0]);
set(gca,'yticklabel',{'-2.0' '' '2.0'}, 'FontSize', 20);
set(gca, 'linewidth', 1.5, 'TickLength', [0.05 0.06], ...
            'FontSize', 24, 'Color', 'none', 'TickDir', 'out', 'Box', 'off');


nexttile;
plot(x, shuffle_signal(4, :), "Color", shuffle_color, 'LineWidth', 2, 'LineStyle', '--');
hold on;
plot(x, signal(4, :), "Color", origin_color, 'LineWidth', 2, 'LineStyle', '--');
hold on;
plot(x, Noattn_predict(4, :), "Color", predict_color, 'LineWidth', 2);
ylim([-3 3]);
yticks([-2.0 0. 2.0]);
set(gca,'yticklabel',{'-2.0' '' '2.0'}, 'FontSize', 20);
set(gca, 'linewidth', 1.5, 'TickLength', [0.05 0.06], ...
            'FontSize', 24, 'Color', 'none', 'TickDir', 'out', 'Box', 'off');
t.TileSpacing = 'compact';
t.Padding = 'compact';
set(gcf,'position',[100,100,1000,500]);

%% head 1&8 plot
x = 1:1:100;
t = tiledlayout(2, 1);

nexttile;
plot(x, shuffle_signal(1, :), "Color", shuffle_color, 'LineWidth', 2, 'LineStyle', '--');
hold on;
plot(x, signal(1, :), "Color", origin_color, 'LineWidth', 2, 'LineStyle', '--');
hold on;
plot(x, predict(1, :), "Color", predict_color, 'LineWidth', 2);
hold on;
plot(x, mask_predict(1, :), "Color", mask_color, 'LineWidth', 2);
ylim([-3 3]);
yticks([-2.0 0. 2.0]);
set(gca,'yticklabel',{'-2.0' '' '2.0'}, 'FontSize', 20);
set(gca, 'linewidth', 1.5, 'TickLength', [0.05 0.06], ...
            'FontSize', 24, 'Color', 'none', 'TickDir', 'out', 'Box', 'off');


nexttile;
plot(x, shuffle_signal(4, :), "Color", shuffle_color, 'LineWidth', 2, 'LineStyle', '--');
hold on;
plot(x, signal(4, :), "Color", origin_color, 'LineWidth', 2, 'LineStyle', '--');
hold on;
plot(x, predict(4, :), "Color", predict_color, 'LineWidth', 2);
hold on;
plot(x, mask_predict(4, :), "Color", mask_color, 'LineWidth', 2);
ylim([-3 3]);
yticks([-2.0 0. 2.0]);
set(gca,'yticklabel',{'-2.0' '' '2.0'}, 'FontSize', 20);
set(gca, 'linewidth', 1.5, 'TickLength', [0.05 0.06], ...
            'FontSize', 24, 'Color', 'none', 'TickDir', 'out', 'Box', 'off');
t.TileSpacing = 'compact';
t.Padding = 'compact';
set(gcf,'position',[100,100,1000,500]);