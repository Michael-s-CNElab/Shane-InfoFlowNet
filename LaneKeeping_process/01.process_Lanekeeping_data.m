%% load and pre-process s05_061019m set

ch_name ={'Fz', 'Cz', 'Pz', 'Oz'};
filename = 's05_061019m.set';
EEG = pop_loadset('filename', filename);
EEG = pop_eegfiltnew(EEG, 'locutoff',0.1,'hicutoff',50,'plotfreqz',0);
EEG = pop_chanedit(EEG, 'lookup','C:\Users\Howard\Documents\MATLAB2021\toolbox\eeglab13_6_5b\plugins\dipfit2.3\standard_BESA\standard-10-5-cap385.elp','load',{'F:\antiCap32.ced','filetype','autodetect'},'rplurchanloc',1,'lookup','C:\Users\Howard\Documents\MATLAB2021\toolbox\eeglab13_6_5b\plugins\dipfit2.3\standard_BESA\standard-10-5-cap385.elp');
EEG = pop_reref( EEG, [] ,'keepref','on');
EEG.data = normalize(double(EEG.data));
EEG = pop_epoch(EEG, {'251','252'}, [-1, 2]);
idx = (EEG.times <= 0);
eeglab redraw;

Data = EEG.data;

s=1;
trial_info = {};
count = 1;
for t=1:683
    if length(EEG.epoch(t).eventlatency) >= 2
        if EEG.epoch(t).eventlatency{2} > 100
            trial_info{end+1, 1} = count;
            trial_info{end, 2} = s;
            trial_info{end, 3} = t;
            trial_info{end, 4} = EEG.epoch(t).eventlatency{2};
            count = count + 1;
        end
    end
end

T = array2table(trial_info);
T.Properties.VariableNames(1:4) = {'trial number', 'subject', 'trial number in EEGset', 'RT'};
writetable(T, 'trial_info.csv');

RT_index = cell2mat(trial_info);

data = ones(4, 1500, 581);

for t=1:581
    data(1, :, t) = EEG.data(5, :, RT_index(t, 3));  % Fz
    data(2, :, t) = EEG.data(15, :, RT_index(t, 3)); % Cz
    data(3, :, t) = EEG.data(25, :, RT_index(t, 3)); % Pz
    data(4, :, t) = EEG.data(31, :, RT_index(t, 3)); % Oz
end

%% 4 channel data save to csv

data_name = 'Lanekeeping';
trial=581;
window_size=100;
overlap=10;
pts = 1500;
subject=1;
samplingrate = 500;
ch_name ={'Fz', 'Cz', 'Pz', 'Oz'};
save_path = strcat("I:\\共用雲端硬碟\\CNElab_枋劭勳\\10.交接資料\\Shane-InfoFlowNet\\data\\", data_name, "\\");

max_count = ((pts-window_size)/overlap)+1;

label={};

WOI = 826:1:1127;

for s=1:subject
    for t=1:trial
        csv_name=strcat(num2str(s, "%02d"), '_', num2str(t, "%03d"), '.csv');
%         trial_data = transpose(data(:, :, t, s));
%         T = array2table(trial_data);
%         T.Properties.VariableNames(1:4) = ch_name;
%         filepathname = strcat(save_path, "csvdata\\", csv_name);
%         writetable(T, filepathname);
        for c=1:max_count

            label{end+1, 1}=s;
            label{end, 2}=t;
    
            win_start_pt = ((c-1)*overlap)+1;
            win_end_pt = win_start_pt+window_size-1;

            label{end, 3}=win_start_pt;
            label{end, 4}=win_end_pt;

            label{end, 5}=win_start_pt / samplingrate;
            label{end, 6}=win_end_pt / samplingrate;

            if find(win_start_pt==WOI) & find(win_end_pt==WOI)
                label{end, 7}=1;
            else
                label{end, 7}=0;
            end
            label{end, 8}=csv_name;

        end
    end
end

T = array2table(label);
T.Properties.VariableNames(1:8) = {'subject' 'trial'...
                    'window start pt' 'window end pt'...
                    'window start time' 'window end time' 'WOI', 'csvname'};
writetable(T, strcat(save_path, data_name, "_overlap10_label.csv"));

%% create WOI baseline label

data_name = 'Lanekeeping';
trial=581;
window_size=100;
overlap=10;
subject=1;
samplingrate = 500;
ch_name ={'Fz', 'Cz', 'Pz', 'Oz'};
save_path = strcat("I:\\共用雲端硬碟\\CNElab_枋劭勳\\10.交接資料\\Shane-InfoFlowNet\\data\\", data_name, "\\");

max_count = ((length(data)-window_size)/overlap)+1;

label={};

WOI = 1:1:500;

for s=1:subject
    for t=1:trial
        csv_name=strcat(num2str(s, "%02d"), '_', num2str(t, "%03d"), '.csv');
        for c=1:max_count

            label{end+1, 1}=s;
            label{end, 2}=t;

            temp = data(:, c:c+window_size-1, t);
    
            win_start_pt = ((c-1)*overlap)+1;
            win_end_pt = win_start_pt+window_size-1;

            label{end, 3}=win_start_pt;
            label{end, 4}=win_end_pt;

            label{end, 5}=win_start_pt / samplingrate;
            label{end, 6}=win_end_pt / samplingrate;

            if find(win_start_pt==WOI) & find(win_end_pt==WOI)
                label{end, 7}=1;
            else
                label{end, 7}=0;
            end
            label{end, 8}=csv_name;

        end
    end
end

T = array2table(label);
T.Properties.VariableNames(1:8) = {'subject' 'trial'...
                    'window start pt' 'window end pt'...
                    'window start time' 'window end time' 'WOI', 'csvname'};
writetable(T, strcat(save_path, data_name, "_baseline_overlap10_label.csv"));
