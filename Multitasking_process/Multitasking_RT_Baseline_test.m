function [Mean, SEM] = Multitasking_RT_Baseline_test(RT_window, BSL_window)
    for w=7:8
            a=zeros(1, 36);
            b=zeros(1, 36);
            for s=1:36
                a(s) = mean(mean(corr2_corr1(c_0, c, w, (s*30)-1:s*30, :), 5), 4);
                b(s) = mean(mean(corr2_corr1(c_0, c, 5, (s*30)-1:s*30, :), 5), 4);
            end
            H_array(1, w-4) = ttest(a, b);
        end
end