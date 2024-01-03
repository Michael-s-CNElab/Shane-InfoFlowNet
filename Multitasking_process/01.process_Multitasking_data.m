%% 4 channel pre-process
ch_name ={'Fz', 'Cz', 'Pz', 'Oz'};
data = ones(4, 1500, 30, 36);
RT = ones(30, 36);
bad_trial_info = {};
trial_count = 0;
trial_info = {};

for s=1:36
    trials = length(EEG_set{s, 1}.epoch);
    count = 0;
    for e=1:trials
        if strcmp(EEG_set{s, 1}.epoch(e).eventtype{end}, '2510')
            count = count + 1;
            trial_count = trial_count + 1;
            RT(count, s) = EEG_set{s, 1}.epoch(e).eventlatency{end};
            
            Data = EEG_set{s, 1}.data(: ,:, e);
            Data = normalize(Data);

            data(1, :, count, s) = Data(2, :);  %Fz
            data(2, :, count, s) = Data(24, :); %Cz
            data(3, :, count, s) = Data(13, :); %Pz
            data(4, :, count, s) = Data(17, :); %Oz
            
            trial_info{end+1, 1} = trial_count;
            trial_info{end, 2} = s;
            trial_info{end, 3} = e;
            trial_info{end, 4} = EEG_set{s, 1}.epoch(e).eventlatency{end};
        else
            bad_trial_info{end+1, 1} = s;
            bad_trial_info{end, 2} = e;
        end
        
        if count == 30
            break;
        end
        
    end
end

T = array2table(trial_info);
T.Properties.VariableNames(1:4) = {'trial number', 'subject', 'trial number in EEGset', 'RT'};
writetable(T, 'trial_info.csv');

%% 4 channel data save to csv

data_name = 'Multitasking';
trial=30;
pts = 1500;
window_size=100;
overlap=10;
subject=36;
samplingrate = 500;
ch_name ={'Fz', 'F3', 'F4', 'Cz', 'C3', 'C4', 'Pz', 'Oz'};
save_path = strcat("I:\\共用雲端硬碟\\CNElab_枋劭勳\\10.交接資料\\Shane-InfoFlowNet\\data\\", data_name, "\\");

max_count = ((pts-window_size)/overlap)+1;

label={};

WOI = 595:1:776;

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

