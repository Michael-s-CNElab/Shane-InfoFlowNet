%% plot signal
path = './plot_signal_data/';
signal = [path, '01_134.csv'];
shuffle_signal = [path, 'head1_01_134_shuffle_signal.csv'];
predict_signal = [path 'head1_01_134_predict.csv'];

Input = readmatrix(signal);
Input = Input(1:100, :);
shuffle = readmatrix(shuffle_signal);
predict = readmatrix(predict_signal);

x = 1:1:100;
t = tiledlayout(4, 1);

for i=1:4
    nexttile;
    plot(x, Input(:, i), 'LineWidth', 3, 'Color', 'g', 'DisplayName', 'origin');
    hold on;
    plot(x, shuffle(:, i), 'LineWidth', 3, 'Color', 'b', 'DisplayName', 'shuffle');
    hold on;
    plot(x, predict(:, i), 'LineWidth', 3, 'Color', 'r', 'DisplayName', 'predict');

    xticks([20 40 60 80]);
    xticklabels({'', '', '', ''});
    ylim([-3 3]);
    yticks([-2 0 2]);
    yticklabels({'-2', '', '2'})
    if i==4
        xticklabels({'20', '40', '60', '80'});
        % legend('Location', 'best');
    end
    set(gca, 'linewidth', 1.5, 'TickLength', [0.05 0.12], ...
            'FontSize', 24, 'Color', 'none', 'Fontname', 'Arial', ...
            'TitleFontWeight', 'normal', 'TickDir', 'out', 'box', 'off');
end
t.Padding = 'compact';
set(gcf,'position',[100,100,1000,500]);

%% attention weight
channel = 4;
head = 8;
window = 15;

t = tiledlayout(head, window);

for h=1:head
    for w=1:window
        nexttile;
        a = squeeze(attention_weight(w, h, :, :));
        imagesc(a);
        axis off;
    end
end
t.Padding = 'compact';
set(gcf,'position',[0,0,1800,head*120]);
