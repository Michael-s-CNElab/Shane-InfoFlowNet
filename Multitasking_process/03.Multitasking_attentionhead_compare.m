windows = 15;
subject = 36;
channel = 4;
trial = 30;
shuffle = 30;
ch_location = {'Fz', 'Cz', 'Pz', 'Oz'};

cos2_cos1_Noattn = ones(channel, channel, windows, trial, subject, shuffle);
cos2_cos1_head1 = ones(channel, channel, windows, trial, subject, shuffle);
cos2_cos1_head8 = ones(channel, channel, windows, trial, subject, shuffle);

for sub=1:subject
    for t=1:trial
        for s=1:shuffle
            for w=1:windows
                for c=1:channel
                    for c_shuffle=1:channel
                        if cos2_Noattn(c, c_shuffle, w, t, sub, s) > cos1_Noattn(c, w, t, s)
                            cos2_Noattn(c, c_shuffle, w, t, sub, s) = cos1_Noattn(c, w, t, s);
                        end

                        if cos2_head1(c, c_shuffle, w, t, sub, s) > cos1_head1(c, w, t, s)
                            cos2_head1(c, c_shuffle, w, t, sub, s) = cos1_head1(c, w, t, s);
                        end
    
                        if cos2_head8(c, c_shuffle, w, t, sub, s) > cos1_head8(c, w, t, s)
                            cos2_head8(c, c_shuffle, w, t, sub, s) = cos1_head8(c, w, t, s);
                        end

                        cos2_cos1_Noattn(c, c_shuffle, w, t, sub, s) = -(cos2_Noattn(c, c_shuffle, w, t, sub, s)-...
                        cos1_Noattn(c, w, t, s));
    
                        cos2_cos1_head1(c, c_shuffle, w, t, sub, s) = -(cos2_head1(c, c_shuffle, w, t, sub, s)-...
                        cos1_head1(c, w, t, s));
    
                        cos2_cos1_head8(c, c_shuffle, w, t, sub, s) = -(cos2_head8(c, c_shuffle, w, t, sub, s)-...
                        cos1_head8(c, w, t, s));
                end
            end
        end
    end
    end
end

[cos2_cos1_Noattn_mean, cos2_cos1_Noattn_sem] = InfoFlowNet_Causality_Rescale(cos2_cos1_Noattn, 10, trial);
[cos2_cos1_head1_mean, cos2_cos1_head1_sem] = InfoFlowNet_Causality_Rescale(cos2_cos1_head1, 10, trial);
[cos2_cos1_head8_mean, cos2_cos1_head8_sem] = InfoFlowNet_Causality_Rescale(cos2_cos1_head8, 10, trial);

x = 1:1:windows;
t = tiledlayout(channel, channel);

for c_0=1:channel
    for c=1:channel
        nexttile;

        Aver = [];
        Var = [];
        for i=1:windows
            Aver(1, end+1) = cos2_cos1_Noattn_mean(c_0, c, i);
            Var(1, end+1) = cos2_cos1_Noattn_sem(c_0, c, i);
        end
        a = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.067 0.494 0.710], 'DisplayName', 'No attention');
        hold on;

        Aver = [];
        Var = [];
        for i=1:windows
            Aver(1, end+1) = cos2_cos1_head1_mean(c_0, c, i);
            Var(1, end+1) = cos2_cos1_head1_sem(c_0, c, i);
        end
        a = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.867 0.094 0.310], 'DisplayName', 'head1');
        hold on;

        Aver = [];
        Var = [];
        for i=1:windows
            Aver(1, end+1) = cos2_cos1_head8_mean(c_0, c, i);
            Var(1, end+1) = cos2_cos1_head8_sem(c_0, c, i);
        end
        a = errorbar(x, Aver, Var, 'LineWidth', 2, 'Color', [0.367 0.894 0.510], 'DisplayName', 'head8');
        hold on;
        
        
        xlim([-1 16]);
        xticks([1 5 10 15]);
        if c_0 == c
            ylim([0.6 1.1]);
            yticks([0.7 0.85 1.0]);
            set(gca,'yticklabel',{'0.7' '' '1.0'}, 'FontSize', 20);
        else
            ylim([-0.05 0.65]);
            yticks([0.0 0.3 0.6]);
            set(gca,'yticklabel',{'0.0' '' '0.6'}, 'FontSize', 20);
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
        legend('Location', 'best');

    end
end
t.TileSpacing = 'compact';
t.Padding = 'compact';
set(gcf,'position',[100,100,1400,700]);

%% plot signal

x = 1:1:100;
signal_data = readtable("plot_signal\signal.csv");
shuffle_data = readtable("plot_signal\shuffle_signal.csv");
pred_data = readtable("plot_signal\predict.csv");

signal_data = readtable("plot_signal\head8_signal.csv");
shuffle_data = readtable("plot_signal\head8_shuffle_signal.csv");
pred_data = readtable("plot_signal\head8_predict.csv");

signal_data = readtable("plot_signal\head0_signal.csv");
shuffle_data = readtable("plot_signal\head0_shuffle_signal.csv");
pred_data = readtable("plot_signal\head0_predict.csv");

t = tiledlayout(2, 1);
nexttile;
plot(shuffle_data.Fz, "LineWidth", 2, "Color", [0.195, 0.447, 0.841], "DisplayName", "shffule");
hold on;
plot(pred_data.Fz, "LineWidth", 2, "Color", [0.895, 0.247, 0.141], "DisplayName", "predict");
hold on;
plot(shuffle_data.Fz, "LineWidth", 2, "Color", [0.295, 0.747, 0.341], "DisplayName", "origin");
ylim([-1.5 0]);
set(gca, 'linewidth', 1.5, 'TickLength', [0.05 0.12], ...
            'FontSize', 24, 'Color', 'none', 'TickDir', 'out', 'Box', 'off');

nexttile;
plot(shuffle_data.Oz, "LineWidth", 2, "Color", [0.195, 0.447, 0.841], "DisplayName", "shffule");
hold on;
plot(pred_data.Oz, "LineWidth", 2, "Color", [0.895, 0.247, 0.141], "DisplayName", "predict");
hold on;
plot(signal_data.Oz, "LineWidth", 2, "Color", [0.295, 0.747, 0.341], "DisplayName", "origin");
ylim([-1.5 1.5]);
set(gca, 'linewidth', 1.5, 'TickLength', [0.05 0.12], ...
            'FontSize', 24, 'Color', 'none', 'TickDir', 'out', 'Box', 'off');
% legend('Location', 'best');

t.TileSpacing = 'compact';
t.Padding = 'compact';
set(gcf,'position',[100,100,1000,500]);

%% plot mask signal
x = 1:1:100;
head1_mask_pred = readtable("plot_signal\head1_mask_predict.csv");
head8_mask_pred = readtable("plot_signal\head8_mask_predict.csv");

t = tiledlayout(2, 1);
nexttile;
plot(head8_mask_pred.Fz, "LineWidth", 2, "Color", [0.995, 0.047, 0.041], "DisplayName", "predict mask");
ylim([-1.5 0]);
yticks([-1.0 -0.5]);
set(gca, 'linewidth', 1.5, 'TickLength', [0.05 0.12], ...
            'FontSize', 24, 'Color', 'none', 'TickDir', 'out', 'Box', 'off');

nexttile;
plot(head8_mask_pred.Oz, "LineWidth", 2, "Color", [0.995, 0.047, 0.041], "DisplayName", "predict mask");
ylim([-1.5 1.5]);
yticks([-1. 0.0 1]);
set(gca, 'linewidth', 1.5, 'TickLength', [0.05 0.12], ...
            'FontSize', 24, 'Color', 'none', 'TickDir', 'out', 'Box', 'off');

t.TileSpacing = 'compact';
t.Padding = 'compact';
set(gcf,'position',[100,100,1000,500]);