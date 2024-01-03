%% read result matrix
windows = 15;
subject = 1;
channel = 4;
trial = 581;
shuffle = 30;
ch_location = {'Fz', 'Cz', 'Pz', 'Oz'};

corr1 = ones(channel, windows, trial, subject);
corr2 = ones(channel, channel, windows, trial, subject, shuffle);
xcorr1 = ones(channel, windows, trial, subject);
xcorr2 = ones(channel, channel, windows, trial, subject, shuffle);
cos1 = ones(channel, windows, trial, subject);
cos2 = ones(channel, channel, windows, trial, subject, shuffle);

mat_path = 'Lanekeeping_head4_result\';
mat_files = dir([mat_path, '*.mat']);

for i=1:length(mat_files)
    name_spilt = split(mat_files(i).name, ["_", "."]);
    sub = str2num(name_spilt{1});
    tr = str2num(name_spilt{2});
    w = round(str2num(name_spilt{3}) / 100) + 1;
    a = load([mat_path, mat_files(i).name]);
    
    corr1(:, w, tr, sub) = a.corr1;
    corr2(:, :, w, tr, sub, :) = a.corr2;
    xcorr1(:, w, tr, sub) = a.xcorr1;
    xcorr2(:, :, w, tr, sub, :) = a.xcorr2;
    cos1(:, w, tr, sub) = a.cos1;
    cos2(:, :, w, tr, sub, :) = a.cos2;

    disp(strcat(mat_files(i).name, ' is done.'));
end

%% read mask result matrix
windows = 15;
subject = 1;
channel = 4;
trial = 581;
shuffle = 30;
ch_location = {'Fz', 'Cz', 'Pz', 'Oz'};

corr1_eye = ones(channel, windows, trial, subject);
corr2_eye = ones(channel, channel, windows, trial, subject, shuffle);
xcorr1_eye = ones(channel, windows, trial, subject);
xcorr2_eye = ones(channel, channel, windows, trial, subject, shuffle);
cos1_eye = ones(channel, windows, trial, subject);
cos2_eye = ones(channel, channel, windows, trial, subject, shuffle);

mat_path = 'Lanekeeping_head4_eye_result\';
mat_files = dir([mat_path, '*.mat']);

for i=1:length(mat_files)
    name_spilt = split(mat_files(i).name, ["_", "."]);
    sub = str2num(name_spilt{1});
    tr = str2num(name_spilt{2});
    w = round(str2num(name_spilt{3}) / 100) + 1;
    a = load([mat_path, mat_files(i).name]);
    
    corr1_eye(:, w, tr, sub) = a.corr1;
    corr2_eye(:, :, w, tr, sub, :) = a.corr2;
    xcorr1_eye(:, w, tr, sub) = a.xcorr1;
    xcorr2_eye(:, :, w, tr, sub, :) = a.xcorr2;
    cos1_eye(:, w, tr, sub) = a.cos1;
    cos2_eye(:, :, w, tr, sub, :) = a.cos2;

    disp(strcat(mat_files(i).name, ' is done.'));
end

%% connectivity result
windows = 15;
subject = 1;
channel = 4;
trial = 581;
shuffle = 30;
ch_location = {'Fz', 'Cz', 'Pz', 'Oz'};

corr2_corr1 = ones(channel, channel, windows, trial, subject, shuffle);
xcorr2_xcorr1 = ones(channel, channel, windows, trial, subject, shuffle);
cos2_cos1 = ones(channel, channel, windows, trial, subject, shuffle);

for sub=1:subject
    for t=1:trial
        for s=1:shuffle
            for w=1:windows
                for c=1:channel
                    for c_shuffle=1:channel
                        if corr2(c, c_shuffle, w, t, sub, s) > corr1(c, w, t)
                            corr2(c, c_shuffle, w, t, sub, s) = corr1(c, w, t);
                        end

                        if xcorr2(c, c_shuffle, w, t, sub, s) > xcorr1(c, w, t)
                            xcorr2(c, c_shuffle, w, t, sub, s) = xcorr1(c, w, t);
                        end
    
                        if cos2(c, c_shuffle, w, t, sub, s) > cos1(c, w, t)
                            cos2(c, c_shuffle, w, t, sub, s) = cos1(c, w, t);
                        end

                        corr2_corr1(c, c_shuffle, w, t, sub, s) = -(corr2(c, c_shuffle, w, t, sub, s)-...
                        corr1(c, w, t));
    
                        xcorr2_xcorr1(c, c_shuffle, w, t, sub, s) = -(xcorr2(c, c_shuffle, w, t, sub, s)-...
                        xcorr1(c, w, t));
    
                        cos2_cos1(c, c_shuffle, w, t, sub, s) = -(cos2(c, c_shuffle, w, t, sub, s)-...
                        cos1(c, w, t));
                    end
                end
            end
        end
    end
end

[corr2_corr1_mean, corr2_corr1_sem] = InfoFlowNet_Causality_Rescale(corr2_corr1, 10, trial);
[xcorr2_xcorr1_mean, xcorr2_xcorr1_sem] = InfoFlowNet_Causality_Rescale(xcorr2_xcorr1, 10, trial);
[cos2_cos1_mean, cos2_cos1_sem] = InfoFlowNet_Causality_Rescale(cos2_cos1, 10, trial);
% [DTW2_DTW1_mean, DTW2_DTW1_sem] = InfoFlowNet_Causality_Rescale(DTW2_DTW1, 10, trial);

x = 1:1:windows;
t = tiledlayout(channel, channel);

for c_0=1:channel
    for c=1:channel
        nexttile;   

        % correlation
%         Aver = [];
%         Var = [];
%         for i=1:windows
%             Aver(1, end+1) = corr2_corr1_mean(c_0, c, i);
%             Var(1, end+1) = corr2_corr1_sem(c_0, c, i);
%         end
%         a = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.067 0.494 0.710], 'DisplayName', 'correlation');
%         hold on;
%         H_array = nan(1, windows);
%         RT_w = squeeze(mean(corr2_corr1(c_0, c, 7, :, :, :), 4));
%         BSL_w = squeeze(mean(corr2_corr1(c_0, c, 1, :, :, :), 4));
%         H_array(7) = ttest2(RT_w, BSL_w);
%         RT_w = squeeze(mean(corr2_corr1(c_0, c, 8, :, :, :), 4));
%         H_array(8) = ttest2(RT_w, BSL_w);
%         RT_w = squeeze(mean(corr2_corr1(c_0, c, 9, :, :, :), 4));
%         H_array(9) = ttest2(RT_w, BSL_w);
        % correlation
        
        % cross-correlation
%         Aver = [];
%         Var = [];
%         for i=1:windows
%             Aver(1, end+1) = xcorr2_xcorr1_mean(c_0, c, i);
%             Var(1, end+1) = xcorr2_xcorr1_sem(c_0, c, i);
%         end
%         b = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.067 0.494 0.710], 'DisplayName', 'cross-correlation');
%         hold on;
%         H_array = nan(1, windows);
%         RT_w = squeeze(mean(xcorr2_xcorr1(c_0, c, 7, :, :, :), 4));
%         BSL_w = squeeze(mean(xcorr2_xcorr1(c_0, c, 1, :, :, :), 4));
%         H_array(7) = ttest2(RT_w, BSL_w);
%         RT_w = squeeze(mean(xcorr2_xcorr1(c_0, c, 8, :, :, :), 4));
%         H_array(8) = ttest2(RT_w, BSL_w);
%         RT_w = squeeze(mean(xcorr2_xcorr1(c_0, c, 9, :, :, :), 4));
%         H_array(9) = ttest2(RT_w, BSL_w);
        % cross-correlation
        
        % cosine
        Aver = [];
        Var = [];
        for i=1:windows
            Aver(1, end+1) = cos2_cos1_mean(c_0, c, i);
            Var(1, end+1) = cos2_cos1_sem(c_0, c, i);
        end
        e = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.067 0.494 0.710], 'DisplayName', 'cosine');
        hold on;
        H_array = nan(1, windows);
        RT_w = squeeze(mean(cos2_cos1(c_0, c, 7, :, :, :), 4));
        BSL_w = squeeze(mean(corr2_corr1(c_0, c, 1, :, :, :), 4));
        H_array(7) = ttest2(RT_w, BSL_w);
        RT_w = squeeze(mean(cos2_cos1(c_0, c, 8, :, :, :), 4));
        H_array(8) = ttest2(RT_w, BSL_w);
        RT_w = squeeze(mean(cos2_cos1(c_0, c, 9, :, :, :), 4));
        H_array(9) = ttest2(RT_w, BSL_w);
        % cosine

%         Aver = [];
%         Var = [];
%         for i=1:windows
%             Aver(1, end+1) = DTW2_DTW1_mean(c_0, c, i);
%             Var(1, end+1) = DTW2_DTW1_sem(c_0, c, i);
%         end
%         f = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.7685 0.7736 0.1825], 'DisplayName', 'DTW');
%         xlim([0 windows+1]);
        
        a = errorbar(x(7:9), Aver(7:9), Var(7:9), 'LineWidth', 4, 'Color', 'r');
        
        xlim([-1 16]);
        xticks([1 5 10 15]);
        if c_0 == c
            ylim([0.3 0.9]);
            yticks([0.5 0.7]);
            set(gca,'yticklabel',{'0.5' '0.7'}, 'FontSize', 20);
            scatter(x, H_array-.1, 300, 'red', '*');
        else
            ylim([0.1 .7]);
            yticks([0.3 0.5]);
            set(gca,'yticklabel',{'0.3' '0.5'}, 'FontSize', 20);
            scatter(x, H_array-.3, 300, 'red', '*');
        end

%         ylim([-0.1 1.1]);
%         yticks([0.0 0.5 1.0]);
%         set(gca,'yticklabel',{'0.0' '' '1.0'}, 'FontSize', 20);
%         scatter(x, H_array, 300, 'red', '*');

        set(gca,'xticklabel',{'' '' '' ''}, 'FontSize', 12);
        if c_0==channel
            set(gca,'xticklabel',{'-1' '0' '1' '2'}, 'FontSize', 20);
        end
        if c_0==1
            title(ch_location(c));
        end
        if c==1
            ylabel(ch_location(c_0));
        end
        set(gca, 'linewidth', 1.5, 'TickLength', [0.05 0.12], ...
            'FontSize', 24, 'Color', 'none', 'TickDir', 'out', 'Box', 'off');
        % legend('Location', 'best');

    end
end
t.TileSpacing = 'compact';
t.Padding = 'compact';
set(gcf,'position',[100,100,1400,700]);

%% Lanekeeping data model attention weight

head = 1;
windows_num = 15;
t = tiledlayout(windows_num, head);
a = mean(attention_weight, 5);
for w=1:windows_num
    for h=1:head
        nexttile;
        clims = [0 1];
        c = colorbar;
        imagesc(squeeze(a(h, :, :, w)), clims);
        xticks([1 2 3]);
        xticklabels({'', '', ''});
        yticks([1 2 3]);
        yticklabels({'', '', ''});
    end
end
t.TileSpacing = 'compact';
t.Padding = 'compact';
set(gcf,'position',[0,0,200,1500]);

colorbar

%% connectivity graph

H_array = zeros(4, 4);

for c_0=1:channel
    for c=1:channel
        RT_w = squeeze(mean(cos2_cos1(c_0, c, 7, :, :, :), 6));
        BSL_w = squeeze(mean(cos2_cos1(c_0, c, 1, :, :, :), 6));
        H_array(c_0, c) = ttest(RT_w, BSL_w);
    end
end

mask = H_array == 0;

weight = squeeze(mean(cos2_cos1_mean(:, :, 7, :, :), 6));
weight(mask) = 0;
weight = weight';
BCC=biChordChart(weight,'Label',ch_location ,'Arrow','on');
BCC=BCC.draw();
BCC.setFont('FontName','Arial','FontSize',36);
BCC.setLabelRadius(1.3);
set(gcf,'position',[100,100,900, 900], 'Color', 'none');

MBC = MyBiChordChart(weight, ch_location);
set(gcf,'position',[100,100,900, 900], 'Color', 'none');

%% connectivity no mask vs mask result
windows = 15;
subject = 1;
channel = 4;
trial = 581;
shuffle = 30;
ch_location = {'Fz', 'Cz', 'Pz', 'Oz'};

corr2_corr1 = ones(channel, channel, windows, trial, subject, shuffle);
xcorr2_xcorr1 = ones(channel, channel, windows, trial, subject, shuffle);
cos2_cos1 = ones(channel, channel, windows, trial, subject, shuffle);

corr2_corr1_eye = ones(channel, channel, windows, trial, subject, shuffle);
xcorr2_xcorr1_eye = ones(channel, channel, windows, trial, subject, shuffle);
cos2_cos1_eye = ones(channel, channel, windows, trial, subject, shuffle);

for sub=1:subject
    for t=1:trial
        for s=1:shuffle
            for w=1:windows
                for c=1:channel
                    for c_shuffle=1:channel
                        if corr2(c, c_shuffle, w, t, sub, s) > corr1(c, w, t)
                            corr2(c, c_shuffle, w, t, sub, s) = corr1(c, w, t);
                        end

                        if xcorr2(c, c_shuffle, w, t, sub, s) > xcorr1(c, w, t)
                            xcorr2(c, c_shuffle, w, t, sub, s) = xcorr1(c, w, t);
                        end
    
                        if cos2(c, c_shuffle, w, t, sub, s) > cos1(c, w, t)
                            cos2(c, c_shuffle, w, t, sub, s) = cos1(c, w, t);
                        end

                        corr2_corr1(c, c_shuffle, w, t, sub, s) = -(corr2(c, c_shuffle, w, t, sub, s)-...
                        corr1(c, w, t));
    
                        xcorr2_xcorr1(c, c_shuffle, w, t, sub, s) = -(xcorr2(c, c_shuffle, w, t, sub, s)-...
                        xcorr1(c, w, t));
    
                        cos2_cos1(c, c_shuffle, w, t, sub, s) = -(cos2(c, c_shuffle, w, t, sub, s)-...
                        cos1(c, w, t));
                    end
                end
            end
        end
    end
end

[corr2_corr1_mean, corr2_corr1_sem] = InfoFlowNet_Causality_Rescale(corr2_corr1, 10, trial);
[xcorr2_xcorr1_mean, xcorr2_xcorr1_sem] = InfoFlowNet_Causality_Rescale(xcorr2_xcorr1, 10, trial);
[cos2_cos1_mean, cos2_cos1_sem] = InfoFlowNet_Causality_Rescale(cos2_cos1, 10, trial);

for sub=1:subject
    for t=1:trial
        for s=1:shuffle
            for w=1:windows
                for c=1:channel
                    for c_shuffle=1:channel
                        if corr2_eye(c, c_shuffle, w, t, sub, s) > corr1_eye(c, w, t)
                            corr2_eye(c, c_shuffle, w, t, sub, s) = corr1_eye(c, w, t);
                        end

                        if xcorr2_eye(c, c_shuffle, w, t, sub, s) > xcorr1_eye(c, w, t)
                            xcorr2_eye(c, c_shuffle, w, t, sub, s) = xcorr1_eye(c, w, t);
                        end
    
                        if cos2_eye(c, c_shuffle, w, t, sub, s) > cos1_eye(c, w, t)
                            cos2_eye(c, c_shuffle, w, t, sub, s) = cos1_eye(c, w, t);
                        end

                        corr2_corr1_eye(c, c_shuffle, w, t, sub, s) = abs(corr2_eye(c, c_shuffle, w, t, sub, s)-...
                        corr1_eye(c, w, t));
    
                        xcorr2_xcorr1_eye(c, c_shuffle, w, t, sub, s) = abs(xcorr2_eye(c, c_shuffle, w, t, sub, s)-...
                        xcorr1_eye(c, w, t));
    
                        cos2_cos1_eye(c, c_shuffle, w, t, sub, s) = abs(cos2_eye(c, c_shuffle, w, t, sub, s)-...
                        cos1_eye(c, w, t));
                    end
                end
            end
        end
    end
end

[corr2_corr1_eye_mean, corr2_corr1_eye_sem] = InfoFlowNet_Causality_Rescale(corr2_corr1_eye, 10, trial);
[xcorr2_xcorr1_eye_mean, xcorr2_xcorr1_eye_sem] = InfoFlowNet_Causality_Rescale(xcorr2_xcorr1_eye, 10, trial);
[cos2_cos1_eye_mean, cos2_cos1_eye_sem] = InfoFlowNet_Causality_Rescale(cos2_cos1_eye, 10, trial);
% [DTW2_DTW1_mean, DTW2_DTW1_sem] = InfoFlowNet_Causality_Rescale(DTW2_DTW1, 10, trial);

x = 1:1:windows;
t = tiledlayout(channel, channel);

for c_0=1:channel
    for c=1:channel
        nexttile;   

        Aver = [];
        Var = [];
        for i=1:windows
            Aver(1, end+1) = cos2_cos1_mean(c_0, c, i);
            Var(1, end+1) = cos2_cos1_sem(c_0, c, i);
        end
        e = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.167 0.594 0.110], 'DisplayName', 'No mask');
        hold on;
        a = errorbar(x(9:11), Aver(9:11), Var(9:11), 'LineWidth', 4, 'Color', 'r');

        hold on;
        Aver = [];
        Var = [];
        for i=1:windows
            Aver(1, end+1) = cos2_cos1_eye_mean(c_0, c, i);
            Var(1, end+1) = cos2_cos1_eye_sem(c_0, c, i);
        end
        e = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.067 0.494 0.810], 'DisplayName', 'Mask');
        hold on;
        a = errorbar(x(9:11), Aver(9:11), Var(9:11), 'LineWidth', 4, 'Color', 'r');

%         H_array = nan(1, windows);
%         BSL_w = squeeze(mean(corr2_corr1_eye(c_0, c, 1, :, :, :), 4));
% 
%         RT_w = squeeze(mean(cos2_cos1_eye(c_0, c, 9, :, :, :), 4));
%         H_array(9) = ttest2(RT_w, BSL_w);
%         RT_w = squeeze(mean(cos2_cos1_eye(c_0, c, 10, :, :, :), 4));
%         H_array(10) = ttest2(RT_w, BSL_w);
%         RT_w = squeeze(mean(cos2_cos1_eye(c_0, c, 11, :, :, :), 4));
%         H_array(11) = ttest2(RT_w, BSL_w);
        
        xlim([-1 16]);
        xticks([1 5 10 15]);
        if c_0 == c
            ylim([0.3 0.9]);
            yticks([0.5 0.7]);
            set(gca,'yticklabel',{'0.5' '0.7'}, 'FontSize', 20);
%             scatter(x, H_array-.15, 300, 'red', '*');
        else
            ylim([0.0 .8]);
            yticks([0.2 0.4 .6]);
            set(gca,'yticklabel',{'0.2' '' '0.6'}, 'FontSize', 20);
%             scatter(x, H_array-.25, 300, 'red', '*');
        end

        set(gca,'xticklabel',{'' '' '' ''}, 'FontSize', 12);
        if c_0==channel
            set(gca,'xticklabel',{'-1' '0' '1' '2'}, 'FontSize', 20);
        end
        if c_0==1
            title(ch_location(c));
        end
        if c==1
            ylabel(ch_location(c_0));
        end
        set(gca, 'linewidth', 1.5, 'TickLength', [0.05 0.12], ...
            'FontSize', 24, 'Color', 'none', 'TickDir', 'out', 'Box', 'off');
        % legend('Location', 'best');

    end
end
t.TileSpacing = 'compact';
t.Padding = 'compact';
set(gcf,'position',[100,100,1400,700]);