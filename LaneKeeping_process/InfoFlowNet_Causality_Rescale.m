function [Mean, SEM] = InfoFlowNet_Causality_Rescale(x, alpha, trial)
    % x(x < 0) = 0;
    x = 1. / (1 + exp(-alpha * x));
    x = (x - 0.5) / 0.5;
    % x(x < 0) = 0;
    Mean = mean(mean(x, 6), 4);
    SEM = mean(std(x, 0, 6) / sqrt(trial), 4);
end