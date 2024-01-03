function result = MVGC_timedomain_GC(X, ntrials, momax)

    nobs      = 1000;   % number of observations per trial
    
    regmode   = '';  % VAR model estimation regression mode ('OLS', 'LWR' or empty for default)
    icregmode = '';  % information criteria regression mode ('OLS', 'LWR' or empty for default)
    
    morder    = 'AIC';  % model order to use ('actual', 'AIC', 'BIC' or supplied numerical value)
    
    acmaxlags = 1000;   % maximum autocovariance lags (empty for automatic calculation)
    
    tstat     = '';     % statistical test for MVGC:  'F' for Granger's F-test (default) or 'chi2' for Geweke's chi2 test
    alpha     = 0.05;   % significance level for significance test
    mhtc      = 'FDR';  % multiple hypothesis test correction (see routine 'significance')
    
    fs        = 100;    % sample rate (Hz)
    fres      = [];     % frequency resolution (empty for automatic calculation)

    nvars = size(X, 1);
    ptic('\n*** tsdata_to_infocrit\n');
    [AIC,BIC,moAIC,moBIC] = tsdata_to_infocrit(X,momax,icregmode);
    ptoc('*** tsdata_to_infocrit took ');

    fprintf('\nbest model order (AIC) = %d\n',moAIC);
    fprintf('best model order (BIC) = %d\n',moBIC);
    
    % Select model order.
    
    if strcmpi(morder,'AIC')
        morder = moAIC;
        fprintf('\nusing AIC best model order = %d\n',morder);
    elseif strcmpi(morder,'BIC')
        morder = moBIC;
        fprintf('\nusing BIC best model order = %d\n',morder);
    else
        fprintf('\nusing specified model order = %d\n',morder);
    end
    
    ptic('\n*** tsdata_to_var... ');
    [A,SIG] = tsdata_to_var(X,morder,regmode);
    ptoc;
    
    assert(~isbad(A),'VAR estimation failed');
    
    % Check for failed regression
    
    assert(~isbad(A),'VAR estimation failed');
    
    ptic('*** var_to_autocov... ');
    [G,info] = var_to_autocov(A,SIG,acmaxlags);
    ptoc;
    
    
    ptic('*** autocov_to_pwcgc... ');
    F = autocov_to_pwcgc(G);
    ptoc;
    
    % Check for failed GC calculation
    
    assert(~isbad(F,false),'GC calculation failed');
    
    % Significance test using theoretical null distribution, adjusting for multiple
    % hypotheses.
    
    pval = mvgc_pval(F,morder,nobs,ntrials,1,1,nvars-2,tstat); % take careful note of arguments!
    sig  = significance(pval,alpha,mhtc);
    cd = mean(F(~isnan(F)));

    fprintf('\ncausal density = %f\n',cd);
    result = F;
end