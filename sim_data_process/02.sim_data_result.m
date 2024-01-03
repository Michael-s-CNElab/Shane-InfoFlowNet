%% read result matrix
windows = 5;
subject = 1;
channel = 3;
trial = 100;
shuffle = 30;
ch_location = {'sine', 'sawtooth', 'random'};

corr1 = ones(channel, windows, trial, subject);
corr2 = ones(channel, channel, windows, trial, subject, shuffle);
xcorr1 = ones(channel, windows, trial, subject);
xcorr2 = ones(channel, channel, windows, trial, subject, shuffle);
cos1 = ones(channel, windows, trial, subject);
cos2 = ones(channel, channel, windows, trial, subject, shuffle);

mat_path = 's13_head1_result\';
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
windows = 5;
subject = 1;
channel = 3;
trial = 100;
shuffle = 30;
ch_location = {'sine', 'sawtooth', 'random'};

corr1_eye = ones(channel, windows, trial, subject);
corr2_eye = ones(channel, channel, windows, trial, subject, shuffle);
xcorr1_eye = ones(channel, windows, trial, subject);
xcorr2_eye = ones(channel, channel, windows, trial, subject, shuffle);
cos1_eye = ones(channel, windows, trial, subject);
cos2_eye = ones(channel, channel, windows, trial, subject, shuffle);

mat_path = 's13_head8_eye_result\';
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

%% simulated data causality result
windows = 5;
subject = 1;
channel = 3;
trial = 100;
shuffle = 30;
ch_location = {'sine', 'sawtooth', 'random'};

corr2_corr1 = ones(channel, channel, windows, trial, subject, shuffle);
xcorr2_xcorr1 = ones(channel, channel, windows, trial, subject, shuffle);
cos2_cos1 = ones(channel, channel, windows, trial, subject, shuffle);
DTW2_DTW1 = ones(channel, channel, windows, trial, subject, shuffle);

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
    
                        if DTW2(c, c_shuffle, w, t, sub, s) < DTW1(c, w, t)
                            DTW2(c, c_shuffle, w, t, sub, s) = DTW1(c, w, t);
                        end

                        corr2_corr1(c, c_shuffle, w, t, sub, s) = -(corr2(c, c_shuffle, w, t, sub, s)-...
                        corr1(c, w, t));
    
                        xcorr2_xcorr1(c, c_shuffle, w, t, sub, s) = -(xcorr2(c, c_shuffle, w, t, sub, s)-...
                        xcorr1(c, w, t));
    
                        cos2_cos1(c, c_shuffle, w, t, sub, s) = -(cos2(c, c_shuffle, w, t, sub, s)-...
                        cos1(c, w, t));
    
                        DTW2_DTW1(c, c_shuffle, w, t, sub, s) = DTW2(c, c_shuffle, w, t, sub, s)-...
                        DTW1(c, w, t);
                    end
                end
            end
        end
    end
end

[corr2_corr1_mean, corr2_corr1_sem] = InfoFlowNet_Causality_Rescale(corr2_corr1, 10, trial);
[xcorr2_xcorr1_mean, xcorr2_xcorr1_sem] = InfoFlowNet_Causality_Rescale(xcorr2_xcorr1, 10, trial);
[cos2_cos1_mean, cos2_cos1_sem] = InfoFlowNet_Causality_Rescale(cos2_cos1, 10, trial);
[DTW2_DTW1_mean, DTW2_DTW1_sem] = InfoFlowNet_Causality_Rescale(DTW2_DTW1, 10, trial);

x = 1:1:windows;
t = tiledlayout(channel, channel);

for c_0=1:channel
    for c=1:channel
        nexttile;
        Aver = [];
        Var = [];
        for i=1:windows
            Aver(1, end+1) = corr2_corr1_mean(c_0, c, i);
            Var(1, end+1) = corr2_corr1_sem(c_0, c, i);
        end
        a = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.067 0.494 0.710], 'DisplayName', 'correlation');
        hold on;

        Aver = [];
        Var = [];
        for i=1:windows
            Aver(1, end+1) = xcorr2_xcorr1_mean(c_0, c, i);
            Var(1, end+1) = xcorr2_xcorr1_sem(c_0, c, i);
        end
        b = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.867 0.094 0.310], 'DisplayName', 'cross-correlation');
        hold on;

        Aver = [];
        Var = [];
        for i=1:windows
            Aver(1, end+1) = cos2_cos1_mean(c_0, c, i);
            Var(1, end+1) = cos2_cos1_sem(c_0, c, i);
        end
        e = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.367 0.894 0.510], 'DisplayName', 'cosine');
        hold on;

%         Aver = [];
%         Var = [];
%         for i=1:windows
%             Aver(1, end+1) = DTW2_DTW1_mean(c_0, c, i);
%             Var(1, end+1) = DTW2_DTW1_sem(c_0, c, i);
%         end
%         f = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.7685 0.7736 0.1825], 'DisplayName', 'DTW');
%         xlim([0 windows+1]);

        xticks([1 2 3 4 5]);
        % sigmoid corr
        ylim([-0.1 1.1]);
        yticks([0.0 0.2]);
        set(gca,'yticklabel',{'0.98' '1.0'}, 'FontSize', 20);
        if c_0 == c
            ylim([0.7 1.1]);
            yticks([0.8 0.9 1.0]);
            set(gca,'yticklabel',{'0.8' '' '1.0'}, 'FontSize', 20);
        elseif (c_0 == 1 & c==2) || (c_0 == 2 & c==1) || (c_0 == 3 & c==1)
            ylim([-0.2 1.2]);
            yticks([0.0 0.5 1.0]);
            set(gca,'yticklabel',{'0.0' '' '1.0'}, 'FontSize', 20);
        elseif c_0 == 1 & c==3
            ylim([-0.1 0.5]);
            yticks([0.1 0.3]);
            set(gca,'yticklabel',{'0.1' '0.3'}, 'FontSize', 20);
        elseif (c_0 == 2 & c==3) || (c_0 == 3 & c==2)
            ylim([-0.1 0.7]);
            yticks([0.1 0.3 0.5]);
            set(gca,'yticklabel',{'0.1' '' '0.5'}, 'FontSize', 20);
        end

        set(gca,'xticklabel',{'' '' '' ''}, 'FontSize', 12);
        if c_0==channel
            set(gca,'xticklabel',{'50' '150' '250' '350' '450'}, 'FontSize', 20);
        end
        if c_0==1
            title(ch_location(c));
        end
        if c==1
            ylabel(ch_location(c_0), FontWeight="bold");
        end
        set(gca, 'linewidth', 1.5, 'TickLength', [0.05 0.12], ...
            'FontSize', 24, 'Color', 'none', 'TickDir', 'out', 'Box', 'off');
        % legend('Location', 'best');

    end
end
t.TileSpacing = 'compact';
t.Padding = 'compact';
set(gcf,'position',[100,100,1400,700]);

%% simulated data causality result with attention mask
windows = 5;
subject = 1;
channel = 3;
trial = 100;
shuffle = 30;
ch_location = {'sine', 'sawtooth', 'random'};

corr2_corr1 = ones(channel, channel, windows, trial, subject, shuffle);
xcorr2_xcorr1 = ones(channel, channel, windows, trial, subject, shuffle);
cos2_cos1 = ones(channel, channel, windows, trial, subject, shuffle);
DTW2_DTW1 = ones(channel, channel, windows, trial, subject, shuffle);

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
   
                        corr2_corr1(c, c_shuffle, w, t, sub, s) = -(corr2_eye(c, c_shuffle, w, t, sub, s)-...
                        corr1_eye(c, w, t));
    
                        xcorr2_xcorr1(c, c_shuffle, w, t, sub, s) = -(xcorr2_eye(c, c_shuffle, w, t, sub, s)-...
                        xcorr1_eye(c, w, t));
    
                        cos2_cos1(c, c_shuffle, w, t, sub, s) = -(cos2_eye(c, c_shuffle, w, t, sub, s)-...
                        cos1_eye(c, w, t));
                    end
                end
            end
        end
    end
end

[corr2_corr1_mean, corr2_corr1_sem] = InfoFlowNet_Causality_Rescale(corr2_corr1, 10, trial);
[xcorr2_xcorr1_mean, xcorr2_xcorr1_sem] = InfoFlowNet_Causality_Rescale(xcorr2_xcorr1, 10, trial);
[cos2_cos1_mean, cos2_cos1_sem] = InfoFlowNet_Causality_Rescale(cos2_cos1, 10, trial);

x = 1:1:windows;
t = tiledlayout(channel, channel);

for c_0=1:channel
    for c=1:channel
        nexttile;
        Aver = [];
        Var = [];
        for i=1:windows
            Aver(1, end+1) = corr2_corr1_mean(c_0, c, i);
            Var(1, end+1) = corr2_corr1_sem(c_0, c, i);
        end
        a = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.067 0.494 0.710], 'DisplayName', 'correlation');
        hold on;

        Aver = [];
        Var = [];
        for i=1:windows
            Aver(1, end+1) = xcorr2_xcorr1_mean(c_0, c, i);
            Var(1, end+1) = xcorr2_xcorr1_sem(c_0, c, i);
        end
        b = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.867 0.094 0.310], 'DisplayName', 'cross-correlation');
        hold on;

        Aver = [];
        Var = [];
        for i=1:windows
            Aver(1, end+1) = cos2_cos1_mean(c_0, c, i);
            Var(1, end+1) = cos2_cos1_sem(c_0, c, i);
        end
        e = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.367 0.894 0.510], 'DisplayName', 'cosine');
        hold on;

%         Aver = [];
%         Var = [];
%         for i=1:windows
%             Aver(1, end+1) = DTW2_DTW1_mean(c_0, c, i);
%             Var(1, end+1) = DTW2_DTW1_sem(c_0, c, i);
%         end
%         f = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.7685 0.7736 0.1825], 'DisplayName', 'DTW');
%         xlim([0 windows+1]);

        xticks([1 2 3 4 5]);
        % sigmoid corr
        ylim([-0.1 1.1]);
        yticks([0.0 0.2]);
        set(gca,'yticklabel',{'0.98' '1.0'}, 'FontSize', 20);
        if c_0 == c
            ylim([-0.2 1.2]);
            yticks([0.0 0.5 1.0]);
            set(gca,'yticklabel',{'0.0' '' '1.0'}, 'FontSize', 20);
        else
            ylim([-0.2 1.2]);
            yticks([0.0 0.5 1.0]);
            set(gca,'yticklabel',{'0.0' '' '1.0'}, 'FontSize', 20);
        end

        set(gca,'xticklabel',{'' '' '' ''}, 'FontSize', 12);
        if c_0==channel
            set(gca,'xticklabel',{'50' '150' '250' '350' '450'}, 'FontSize', 20);
        end
        if c_0==1
            title(ch_location(c));
        end
        if c==1
            ylabel(ch_location(c_0), FontWeight="bold");
        end
        set(gca, 'linewidth', 1.5, 'TickLength', [0.05 0.12], ...
            'FontSize', 24, 'Color', 'none', 'TickDir', 'out', 'Box', 'off');
        % legend('Location', 'best');

    end
end
t.TileSpacing = 'compact';
t.Padding = 'compact';
set(gcf,'position',[100,100,1400,700]);

%% simulate data model attention weight

head = 8;
windows_num = 5;
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
set(gcf,'position',[100,100,head*100,500]);

colorbar

%% InfoFlowNet vs GC vs TCDF

x = 1:1:windows;
t = tiledlayout(channel, channel);

for c_0=1:channel
    for c=1:channel
        nexttile;
%         Aver = [];
%         Var = [0 0 0 0 0];
%         for i=1:windows
%             Aver(1, end+1) = MVGC_result(c_0, c, i);
%         end
%         a = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.467 0.894 0.610], 'DisplayName', 'GC', 'LineStyle','-.');
%         hold on;

%         Aver = [];
%         Var = [];
%         for i=1:windows
%             Aver(1, end+1) = att_score_mean(c_0, c, i);
%             Var(1, end+1) = att_score_sem(c_0, c, i);
%         end
%         b = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.267 0.294 0.910], 'DisplayName', 'TCDF', 'LineStyle','--');
%         hold on;
% 
        Aver = [];
        Var = [];
        for i=1:windows
            Aver(1, end+1) = cos2_cos1_mean(c_0, c, i);
            Var(1, end+1) = cos2_cos1_sem(c_0, c, i);
        end
        e = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.967 0.194 0.210], 'DisplayName', 'InfoFlowNet');
        hold on;

%         Aver = [];
%         Var = [];
%         for i=1:windows
%             Aver(1, end+1) = DTW2_DTW1_mean(c_0, c, i);
%             Var(1, end+1) = DTW2_DTW1_sem(c_0, c, i);
%         end
%         f = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.7685 0.7736 0.1825], 'DisplayName', 'DTW');
%         xlim([0 windows+1]);

        xticks([1 2 3 4 5]);
        % sigmoid corr
        ylim([-0.1 0.7]);
        yticks([0.1 0.3 0.5]);
        set(gca,'yticklabel',{'.1' '0.3' '.5'}, 'FontSize', 20);

        set(gca,'xticklabel',{'' '' '' ''}, 'FontSize', 12);
        if c_0==channel
            set(gca,'xticklabel',{'50' '150' '250' '350' '450'}, 'FontSize', 20);
        end
        if c_0==1
            title(ch_location(c));
        end
        if c==1
            ylabel(ch_location(c_0), FontWeight="bold");
        end
        set(gca, 'linewidth', 1.5, 'TickLength', [0.05 0.12], ...
            'FontSize', 24, 'Color', 'none', 'TickDir', 'out', 'Box', 'off');
        % legend('Location', 'best');

    end
end
t.TileSpacing = 'compact';
t.Padding = 'compact';
set(gcf,'position',[100,100,1400,700]);

%% connectivity no mask vs mask result
windows = 5;
subject = 1;
channel = 3;
trial = 100;
shuffle = 30;
ch_location = {'sine', 'sawtooth', 'random'};

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

                        corr2_corr1_eye(c, c_shuffle, w, t, sub, s) = -(corr2_eye(c, c_shuffle, w, t, sub, s)-...
                        corr1_eye(c, w, t));
    
                        xcorr2_xcorr1_eye(c, c_shuffle, w, t, sub, s) = -(xcorr2_eye(c, c_shuffle, w, t, sub, s)-...
                        xcorr1_eye(c, w, t));
    
                        cos2_cos1_eye(c, c_shuffle, w, t, sub, s) = -(cos2_eye(c, c_shuffle, w, t, sub, s)-...
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
        % cosine
        Aver = [];
        Var = [];
        for i=1:windows
            Aver(1, end+1) = cos2_cos1_mean(c_0, c, i);
            Var(1, end+1) = cos2_cos1_sem(c_0, c, i);
        end
        e = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.167 0.594 0.110], 'DisplayName', 'No mask');
        hold on;
        a = errorbar(x(3), Aver(3), Var(3), 'LineWidth', 4, 'Color', 'r');

        hold on;
        Aver = [];
        Var = [];
        for i=1:windows
            Aver(1, end+1) = cos2_cos1_eye_mean(c_0, c, i);
            Var(1, end+1) = cos2_cos1_eye_sem(c_0, c, i);
        end
        e = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.067 0.494 0.810], 'DisplayName', 'Mask');
        hold on;
        a = errorbar(x(3), Aver(3), Var(3), 'LineWidth', 4, 'Color', 'r');
        
        
        if c_0 == c
            ylim([-0.2 1.2]);
            yticks([0.0 0.5 1.0]);
            set(gca,'yticklabel',{'0.0' '' '1.0'}, 'FontSize', 20);
        else
            ylim([-0.2 1.2]);
            yticks([0.0 0.5 1.0]);
            set(gca,'yticklabel',{'0.0' '' '1.0'}, 'FontSize', 20);
        end
        
        xlim([0.5 5.5]);
        xticks([1 2 3 4 5]);
        set(gca,'xticklabel',{'' '' '' ''}, 'FontSize', 12);
        if c_0==channel
            set(gca,'xticklabel',{'50' '150' '250' '350' '450'}, 'FontSize', 20);
        end
        if c_0==1
            title(ch_location(c));
        end
        if c==1
            ylabel(ch_location(c_0), FontWeight="bold");
        end
        set(gca, 'linewidth', 1.5, 'TickLength', [0.05 0.12], ...
            'FontSize', 24, 'Color', 'none', 'TickDir', 'out', 'Box', 'off');
        % legend('Location', 'best');

    end
end
t.TileSpacing = 'compact';
t.Padding = 'compact';
set(gcf,'position',[100,100,1400,700]);