%% gen label
train_info = readmatrix('trial_info.csv');
save_path = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\" + ...
    "data\Lanekeeping_RT_connectivity_feature\head4_csvdata\";

LK_RT_sort = sortrows(train_info, 4);

LK_trial_num = length(train_info);

G1_trial_RT_sort = LK_RT_sort(round(LK_trial_num*0.05):round(LK_trial_num*0.15), :);
G2_trial_RT_sort = LK_RT_sort(round(LK_trial_num*0.85):round(LK_trial_num*0.95), :);


G1_mean = mean(G1_trial_RT_sort);
G2_mean = mean(G2_trial_RT_sort);
G1_std = std(G1_trial_RT_sort);
G2_std = std(G2_trial_RT_sort);

G1_label = sortrows(G1_trial_RT_sort, 1);
G2_label = sortrows(G2_trial_RT_sort, 1);

G1_label(:, 5) = 0;
G2_label(:, 5) = 1;

label = [G1_label; G2_label];

T = array2table(label);
T.Properties.VariableNames(1:5) = {'trial number' 'subject' 'trial' 'RT' 'label'};
writetable(T, strcat(save_path, "LK_G1G2_RT_label.csv"));

%% gen InfoFlowNet connectivity feature
label = table2array(label);
save_path = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\" + ...
    "data\Lanekeeping_RT_connectivity_feature\head2_eye_csvdata\";
baseline_window=5;
cos_feature = squeeze(mean(cos2_cos1, 6));

for t=1:length(label)
    for b=1:baseline_window
        feature = cos_feature(:, :, b, label(t, 1));
        csvname = strcat(num2str(label(t, 2), "%02d"), ...
            "_", num2str(label(t, 1), "%03d"), "_", ...
            num2str(b, "%02d"), ".csv");
        savename = strcat(save_path, csvname);
        writematrix(feature, savename);
    end
end

%% G1 G2 RT line
save_path = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\" + ...
    "data\Lanekeeping_RT_connectivity_feature\head4_csvdata\";
train_info = readmatrix('trial_info.csv');
LK_RT_sort = sortrows(train_info, 4);

LK_trial_num = length(train_info);

plot(1:1:height(LK_RT_sort), LK_RT_sort(:, 4), 'LineWidth', 3);

xticks([200 400]);
set(gca,'xticklabel',{'200' '400'});

yticks([600 1000 1400 1800]);
set(gca,'yticklabel',{'0.6' '1.0' '1.4' '1.8'});

set(gca, 'linewidth', 1.5, 'TickLength', [0.05 0.12], ...
            'FontSize', 24, 'Color', 'none', 'Fontname', 'Arial', ...
            'TitleFontWeight', 'normal', 'TickDir', 'out', 'box', 'off');
xticks([round(LK_trial_num*0.05) round(LK_trial_num*0.15) 200 400 ...
    round(LK_trial_num*0.85) round(LK_trial_num*0.95)]);
set(gca,'xticklabel',{'' '' '200' '400' '' ''});

yticks([600 LK_RT_sort(round(LK_trial_num*0.05), 4) LK_RT_sort(round(LK_trial_num*0.15), 4) 1000 ...
    LK_RT_sort(round(LK_trial_num*0.85), 4) 1400 LK_RT_sort(round(LK_trial_num*0.95), 4) 1800]);
set(gca,'yticklabel',{'0.6' '' '' '1.0' '' '1.4' '' '1.8'});

set(gcf,'position',[100,100,700,600]);


%% G1 G2 connectivity value test
save_path = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\" + ...
    "data\Lanekeeping_RT_connectivity_feature\";
label = readtable(strcat(save_path, "LK_G1G2_RT_label.csv"));
G1_data = zeros(4, 4, 5, 59);
G2_data = zeros(4, 4, 5, 59);

G1_count = 1;
G2_count = 1;

cos2_cos1_m = squeeze(mean(cos2_cos1, 6));

for i=1:height(label)
    tN = label.trialNumber(i);
    G = label.label(i);
    if G == 0
        G1_data(:, :, :, G1_count) = cos2_cos1_m(:, :, 1:5, tN);
        G1_count = G1_count + 1;
    else if G == 1
        G2_data(:, :, :, G2_count) = cos2_cos1_m(:, :, 1:5, tN);
        G2_count = G2_count + 1;
    end
    end
end

x=1:1:2;

t = tiledlayout(channel, channel);
for c_0=1:channel
    for c=1:channel
        nexttile;
        tmp = zeros(2, 59*5);
        tmp(1, :) = squeeze(reshape(G1_data(c_0, c, :, :), [], 1))';
        tmp(2, :) = squeeze(reshape(G2_data(c_0, c, :, :), [], 1))';
        
        mean_data = mean(tmp, 2);
        std_data = std(tmp, 0, 2) / sqrt(59);

        b = bar(mean_data);
        hold on;
        errorbar(x, mean_data, std_data, 'k', 'linestyle', 'none');

        h = ttest(tmp(1, :), tmp(2, :));

        if c_0 == c
            ylim([0.0 0.7]);
            yticks([0.1 0.3 0.5]);
            set(gca,'yticklabel',{'0.1' '' '0.5'}, 'FontSize', 24);
            hold on;
            scatter([1.5], (h*0.8)-0.1, 300, 'red', '*');
        else
            ylim([0.0 0.3]);
            yticks([0.1 0.2]);
            set(gca,'yticklabel',{'0.1' '0.2'}, 'FontSize', 24);
            hold on;
            scatter([1.5], (h*0.4)-0.1, 300, 'red', '*');
        end
        xticks([1 2]);
        set(gca,'xticklabel',{'' ''}, 'FontSize', 24);
        if c_0==4
            xticks([1 2]);
            set(gca,'xticklabel',{'G1' 'G2'}, 'FontSize', 24);
        end
        set(gca, 'linewidth', 1.5, 'TickLength', [0.05 0.12], ...
            'FontSize', 24, 'Color', 'none', 'Fontname', 'Arial', ...
            'TitleFontWeight', 'normal', 'TickDir', 'out', 'box', 'off');
    end
end

t.Padding = 'compact';
set(gcf,'position',[100,100,1400,700]);

%% DL label

train_info = readmatrix('trial_info.csv');
save_path = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\";

LK_RT_sort = sortrows(train_info, 4);

LK_trial_num = length(train_info);

G1_trial_RT_sort = LK_RT_sort(round(LK_trial_num*0.05):round(LK_trial_num*0.15), :);
G2_trial_RT_sort = LK_RT_sort(round(LK_trial_num*0.85):round(LK_trial_num*0.95), :);


G1_mean = mean(G1_trial_RT_sort);
G2_mean = mean(G2_trial_RT_sort);
G1_std = std(G1_trial_RT_sort);
G2_std = std(G2_trial_RT_sort);

G1_label = {};

for tr=1:59
    for w=1:5
        csvname = strcat(num2str(G1_trial_RT_sort(tr, 2), "%02d"), ...
        "_", num2str(G1_trial_RT_sort(tr, 1), "%03d"), ...
        "_", num2str(w, "%02d"), ".csv");
        
        G1_label{end+1, 1} = G1_trial_RT_sort(tr, 1);
        G1_label{end, 2} = G1_trial_RT_sort(tr, 2);
        G1_label{end, 3} = G1_trial_RT_sort(tr, 3);
        G1_label{end, 4} = G1_trial_RT_sort(tr, 4);
        G1_label{end, 5} = csvname;
        G1_label{end, 6} = 0;
    end
end

G2_label = {};

for tr=1:59
    for w=1:5
        csvname = strcat(num2str(G2_trial_RT_sort(tr, 2), "%02d"), ...
        "_", num2str(G2_trial_RT_sort(tr, 1), "%03d"), ...
        "_", num2str(w, "%02d"), ".csv");
        
        G2_label{end+1, 1} = G2_trial_RT_sort(tr, 1);
        G2_label{end, 2} = G2_trial_RT_sort(tr, 2);
        G2_label{end, 3} = G2_trial_RT_sort(tr, 3);
        G2_label{end, 4} = G2_trial_RT_sort(tr, 4);
        G2_label{end, 5} = csvname;
        G2_label{end, 6} = 1;
    end
end

label = [G1_label; G2_label];

T = array2table(label);
T.Properties.VariableNames(1:6) = {'trial number' 'subject' 'trial' 'RT' 'filename' 'label'};
writetable(T, strcat(save_path, "LK_G1G2_RT_DL_label.csv"));

%% DL label for 黃博士

ch_name ={'Fz', 'Cz', 'Pz', 'Oz'};
train_info = readmatrix('trial_info.csv');
save_path = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\data\Lanekeeping_raw_baseline\";

LK_RT_sort = sortrows(train_info, 4);

LK_trial_num = length(train_info);

G1_trial_RT_sort = LK_RT_sort(round(LK_trial_num*0.05):round(LK_trial_num*0.15), :);
G2_trial_RT_sort = LK_RT_sort(round(LK_trial_num*0.85):round(LK_trial_num*0.95), :);


G1_mean = mean(G1_trial_RT_sort);
G2_mean = mean(G2_trial_RT_sort);
G1_std = std(G1_trial_RT_sort);
G2_std = std(G2_trial_RT_sort);

G1_label = {};

for tr=1:59
    csvname = strcat(num2str(G1_trial_RT_sort(tr, 2), "%02d"), ...
        "_", num2str(G1_trial_RT_sort(tr, 3), "%03d"), ".csv");

    T = array2table(data(:, 1:500, G1_trial_RT_sort(tr, 1))');
    T.Properties.VariableNames(1:4) = ch_name;
    writetable(T, strcat(save_path, csvname));
        
    G1_label{end+1, 1} = G1_trial_RT_sort(tr, 2);
    G1_label{end, 2} = G1_trial_RT_sort(tr, 1);
    G1_label{end, 3} = G1_trial_RT_sort(tr, 4);
    G1_label{end, 4} = csvname;
    G1_label{end, 5} = 0;
end

G2_label = {};

for tr=1:59
    csvname = strcat(num2str(G2_trial_RT_sort(tr, 2), "%02d"), ...
        "_", num2str(G2_trial_RT_sort(tr, 3), "%03d"), ".csv");

    T = array2table(data(:, 1:500, G2_trial_RT_sort(tr, 1))');
    T.Properties.VariableNames(1:4) = ch_name;
    writetable(T, strcat(save_path, csvname));
        
    G2_label{end+1, 1} = G2_trial_RT_sort(tr, 2);
    G2_label{end, 2} = G2_trial_RT_sort(tr, 1);
    G2_label{end, 3} = G2_trial_RT_sort(tr, 4);
    G2_label{end, 4} = csvname;
    G2_label{end, 5} = 1;
end

label = [G1_label; G2_label];

T = array2table(label);
T.Properties.VariableNames(1:5) = {'subject' 'trial' 'RT' 'filename' 'label'};
writetable(T, strcat(save_path, "label.csv"));

%% DL label for 黃博士 spilt

ch_name ={'Fz', 'Cz', 'Pz', 'Oz'};
train_info = readmatrix('trial_info.csv');
save_path = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\data\Lanekeeping_raw_baseline_spilt\";

LK_RT_sort = sortrows(train_info, 4);

LK_trial_num = length(train_info);

G1_trial_RT_sort = LK_RT_sort(round(LK_trial_num*0.05):round(LK_trial_num*0.15), :);
G2_trial_RT_sort = LK_RT_sort(round(LK_trial_num*0.85):round(LK_trial_num*0.95), :);


G1_mean = mean(G1_trial_RT_sort);
G2_mean = mean(G2_trial_RT_sort);
G1_std = std(G1_trial_RT_sort);
G2_std = std(G2_trial_RT_sort);

G1_label = {};

for tr=1:59
    for w=1:5
        csvname = strcat(num2str(G1_trial_RT_sort(tr, 2), "%02d"), ...
            "_", num2str(G1_trial_RT_sort(tr, 3), "%03d"), "_", num2str(w, "%02d"), ".csv");
    
        T = array2table(data(:, (w*100)-99:(w*100), G1_trial_RT_sort(tr, 1))');
        T.Properties.VariableNames(1:4) = ch_name;
        writetable(T, strcat(save_path, csvname));
            
        G1_label{end+1, 1} = G1_trial_RT_sort(tr, 2);
        G1_label{end, 2} = G1_trial_RT_sort(tr, 1);
        G1_label{end, 3} = G1_trial_RT_sort(tr, 4);
        G1_label{end, 4} = csvname;
        G1_label{end, 5} = 0;
    end
end

G2_label = {};

for tr=1:59
    for w=1:5
        csvname = strcat(num2str(G2_trial_RT_sort(tr, 2), "%02d"), ...
            "_", num2str(G2_trial_RT_sort(tr, 3), "%03d"), "_", num2str(w, "%02d"), ".csv");
    
        T = array2table(data(:, (w*100)-99:(w*100), G2_trial_RT_sort(tr, 1))');
        T.Properties.VariableNames(1:4) = ch_name;
        writetable(T, strcat(save_path, csvname));
            
        G2_label{end+1, 1} = G2_trial_RT_sort(tr, 2);
        G2_label{end, 2} = G2_trial_RT_sort(tr, 1);
        G2_label{end, 3} = G2_trial_RT_sort(tr, 4);
        G2_label{end, 4} = csvname;
        G2_label{end, 5} = 1;
    end
end

label = [G1_label; G2_label];

T = array2table(label);
T.Properties.VariableNames(1:5) = {'subject' 'trial' 'RT' 'filename' 'label'};
writetable(T, strcat(save_path, "label.csv"));
