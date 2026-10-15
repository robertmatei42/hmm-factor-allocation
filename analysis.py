"""Regime-based allocation between momentum and value.

Course data cover companies in Belgium, France and the Netherlands and are
not distributed with this repository. Set HMM_DATA_DIR or supply files in data/.
Factor spreads are excess returns; total returns include RF. Costs are an
allocation-change proxy. See README.md for methodology and limitations.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# Configuration
DATA_DIR = Path(os.environ.get('HMM_DATA_DIR', str(Path(__file__).resolve().parent / 'data')))
MONTHLY_FILE = DATA_DIR / 'NL_FR_BE_data_monthly.csv'
ANNUAL_FILE = DATA_DIR / 'NL_FR_BE_data_annual.csv'
FACTOR_FILE = DATA_DIR / 'Europe_FF_Factors.csv'
OUTPUT_DIR = Path(__file__).resolve().parent / 'hmm_analysis_results'
COST_PER_UNIT_TRADED = 0.004
COV_TYPE = 'HC0'
HAC_LAGS = 3
SHOW_PLOTS = os.environ.get('HMM_SHOW_PLOTS', '1') == '1'
HEURISTIC_NAME = 'Volatility rule 30-70'


def volatility_benchmark(lagged_volatility, training_months):
    """Freeze the median on the initial training sample; use only lagged inputs."""
    threshold = float(lagged_volatility.iloc[:training_months].median())
    weights = pd.Series(np.where(lagged_volatility > threshold,
            0.7,
            0.3),
        index=lagged_volatility.index)
    weights.iloc[:training_months] = 0.5
    return (weights, threshold)


def assign_deciles(values, bins=10):
    """Assign monthly quantile labels; tied cut points may reduce the number of bins."""
    if values.isna().all():
        return pd.Series(index=values.index, dtype=object)
    ranks = pd.qcut(values, bins, labels=False, duplicates='drop') + 1
    return ranks.map(lambda v: 'p' + str(int(v)).zfill(2) if pd.notna(v) else np.nan)


def wealth_and_drawdown(returns):
    """Compound total returns and measure drawdowns relative to initial wealth of one."""
    values = np.asarray(returns, dtype=float)
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError('Compounding requires finite total returns greater than -100%.')
    wealth = np.cumprod(1 + values)
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return (wealth, wealth / peaks - 1)


def ratio(numerator, denominator):
    """Return NaN when the risk denominator is zero or negative."""
    return numerator / denominator if denominator > 0 else np.nan


def fit_capm(y, market):
    """Regress monthly excess returns on market excess returns with robust inference."""
    X = sm.add_constant(pd.DataFrame({'MktRF': market}), has_constant='add')
    kwargs = {'maxlags': HAC_LAGS} if COV_TYPE == 'HAC' else {}
    return sm.OLS(y, X).fit(cov_type=COV_TYPE, cov_kwds=kwargs)


def performance(returns, rf, market):
    """Calculate excess-return metrics and total-return drawdown; return a CAPM report."""
    excess_returns = returns
    capm = fit_capm(excess_returns, market)
    _, dd = wealth_and_drawdown(rf + returns)
    return ({'Annual_alpha': 12 * capm.params['const'],
            'Alpha_pvalue': capm.pvalues['const'],
            'CAPM_beta': capm.params['MktRF'],
            'Beta_pvalue': capm.pvalues['MktRF'],
            'Appraisal_ratio': np.sqrt(12) * ratio(capm.params['const'],
                capm.resid.std(ddof=1)),
            'Volatility_annual': np.sqrt(12) * returns.std(ddof=1),
            'Sharpe': np.sqrt(12) * ratio(excess_returns.mean(),
                excess_returns.std(ddof=1)),
            'Max_drawdown': dd.min(),
            'Skewness': returns.skew(),
            'Kurtosis_Pearson': returns.kurt() + 3},
        capm.summary().as_text())


def build_factor_inputs():
    """Build calendar-lagged momentum/value spreads and lagged market-state inputs. Annual data are assumed available the following July."""
    data = pd.read_csv(MONTHLY_FILE)
    annual = pd.read_csv(ANNUAL_FILE)
    data['mdate'] = pd.to_datetime(data['mdate'], format='%Y%m')
    data = data.sort_values(['ISIN', 'mdate'])
    data['calendar_month'] = data['mdate'].dt.to_period('M')

    # Exact calendar lags
    me_lookup = data[['ISIN', 'calendar_month', 'ME']].copy()
    me_lookup['calendar_month'] += 1

    # Exact calendar lags
    me_lookup = me_lookup.rename(columns={'ME': 'ME_lag'})

    mom_lookup = data[['ISIN', 'calendar_month', 'RET11']].copy()
    mom_lookup['calendar_month'] += 2

    mom_lookup = mom_lookup.rename(columns={'RET11': 'RET11_lag2'})
    data = data.merge(me_lookup, on=['ISIN', 'calendar_month'], how='left', validate='one_to_one')
    data = data.merge(mom_lookup, on=['ISIN', 'calendar_month'], how='left', validate='one_to_one')
    data = data.sort_values(['ISIN', 'mdate'])

    # Momentum portfolios
    momentum_data = data.dropna(subset=['RET11_lag2', 'ME_lag', 'RET']).copy()
    momentum_data = momentum_data[momentum_data['ME_lag'] > 0].copy()
    momentum_data['momport'] = momentum_data.groupby('mdate')['RET11_lag2'].transform(assign_deciles)
    momentum_data = momentum_data.dropna(subset=['momport'])
    momentum_data['weighted_ret'] = momentum_data['RET'] * momentum_data['ME_lag']

    grouped = momentum_data.groupby(['mdate', 'momport'])[['weighted_ret', 'ME_lag']].sum()
    momentum = (grouped['weighted_ret'] / grouped['ME_lag']).unstack('momport').reset_index()
    momentum['MOM'] = momentum['p10'] - momentum['p01']
    all_portfolios = momentum[['mdate', 'MOM']].copy()

    # Annual accounting data
    annual['fyear'] = annual['fyear'].astype(int)
    annual['avail_date'] = pd.to_datetime((annual['fyear'] + 1).astype(str) + '-07-01')
    annual = annual.sort_values(['avail_date'])
    data = pd.merge_asof(data.sort_values('mdate'),
        annual[['ISIN',
                'avail_date',
                'BEME',
                'OP',
                'INV']],
        left_on='mdate',
        right_on='avail_date',
        by='ISIN',
        direction='backward')
    data = data.drop(columns=['avail_date'])
    data = data.sort_values(['ISIN', 'mdate'])

    # July value portfolios
    data['month'] = data['mdate'].dt.month
    july_data = data[data['month'] == 7].copy()
    july_data['valport'] = july_data.groupby('mdate')['BEME'].transform(assign_deciles)
    july_assignments = july_data[['ISIN',
            'mdate',
            'valport']].rename(columns={'mdate': 'assign_date'})
    data = pd.merge_asof(data.sort_values('mdate'),
        july_assignments.sort_values('assign_date'),
        left_on='mdate',
        right_on='assign_date',
        by='ISIN',
        direction='backward')
    data = data.drop(columns=['assign_date'])
    tmp = data.dropna(subset=['valport', 'ME_lag', 'RET']).copy()
    tmp = tmp.loc[tmp['ME_lag'] > 0].copy()
    tmp['weighted_ret'] = tmp['RET'] * tmp['ME_lag']
    value = tmp.groupby(['mdate', 'valport'])[['weighted_ret', 'ME_lag']].sum()
    value['RET_vw'] = value['weighted_ret'] / value['ME_lag']
    value = value['RET_vw'].unstack().reset_index()
    value['VALUE'] = value['p10'] - value['p01']
    all_portfolios = all_portfolios.merge(value[['mdate', 'VALUE']], on='mdate', how='outer')
    factors = all_portfolios[['mdate', 'MOM', 'VALUE']].copy()
    factors = factors.dropna(subset=['MOM', 'VALUE'])
    factors = factors.sort_values('mdate')

    # Lagged regime inputs
    regime = data.groupby('mdate').agg(market_ret=('RET',
            'mean'),
        avg_ivol=('ivol',
            'mean'),
        dispersion=('RET',
            'std'),
        number=('RET',
            'count')).reset_index()
    regime[['market_ret',
            'avg_ivol',
            'dispersion']] = regime[['market_ret',
                'avg_ivol',
                'dispersion']].shift(1)
    regime = regime.dropna().reset_index(drop=True)
    reg = regime.copy().reset_index(drop=True)
    return (factors, reg)


def estimate_regimes(reg):
    """Fit expanding-window HMMs every 12 months after 36 training observations. Scaling and state labels use training data only. Retain the final posterior of each historical prefix to avoid using future observations; warm-up weights stay neutral."""
    X_all = reg[['market_ret', 'avg_ivol', 'dispersion']].astype(float).values
    T = len(reg)
    n_states = 2
    refit_every = 12
    min_train_months = 36
    if T <= min_train_months:
        raise ValueError('Need more than 36 usable months for HMM evaluation.')
    if not reg['mdate'].dt.to_period('M').astype('int64').diff().dropna().eq(1).all():
        raise ValueError('Regime input has calendar gaps; inspect before fitting.')

    # HMM fit diagnostics
    fit_diagnostics = []
    p_bad = np.full(T, np.nan)
    state_ml = np.full(T, np.nan)

    refit_idx = list(range(min_train_months - 1, T, refit_every))
    if refit_idx[-1] != T - 1:
        refit_idx.append(T - 1)

    for j in range(len(refit_idx) - 1):
        train_end = refit_idx[j]
        pred_end = refit_idx[j + 1]
        X_train = X_all[:train_end + 1]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        hmm = GaussianHMM(n_components=n_states,
            covariance_type='full',
            n_iter=500,
            random_state=42).fit(X_train_s)

        train_states = hmm.predict(X_train_s)
        tmp = pd.DataFrame({'state': train_states,
                'avg_ivol': reg.loc[:train_end,
                    'avg_ivol'].values,
                'dispersion': reg.loc[:train_end,
                    'dispersion'].values})
        state_score = tmp.groupby('state')[['avg_ivol', 'dispersion']].mean().sum(axis=1)
        if len(state_score) != 2:
            raise ValueError('A fitted model did not decode both states; inspect the fit.')
        bad_state = int(state_score.idxmax())
        history = list(hmm.monitor_.history)
        fit_diagnostics.append({'training_end_label': reg.loc[train_end,
                    'mdate'],
                'training_rows': len(X_train),
                'bad_state': bad_state,
                'iterations': hmm.monitor_.iter,
                'converged_flag': hmm.monitor_.converged,
                'last_loglik_change': history[-1] - history[-2] if len(history) > 1 else np.nan,
                'last_loglik': history[-1] if history else np.nan,
                'score_state_0': state_score.get(0,
                    np.nan),
                'score_state_1': state_score.get(1,
                    np.nan)})
        print(f"Fitted HMM through {reg.loc[train_end, 'mdate']:%Y-%m}", flush=True)

        for i in range(train_end + 1, pred_end + 1):
            X_up_to_i = X_all[:i + 1]
            X_up_to_i_s = scaler.transform(X_up_to_i)
            prob = hmm.predict_proba(X_up_to_i_s)
            p_bad[i] = prob[-1, bad_state]
            state_ml[i] = prob[-1].argmax()

    # Probabilities and warm-up allocations
    reg['p_bad'] = p_bad
    reg['state'] = state_ml
    reg['is_warmup'] = np.arange(T) < min_train_months
    if reg.loc[~reg['is_warmup'], 'p_bad'].isna().any():
        raise ValueError('Missing HMM probabilities after warm-up.')
    reg['p_bad'] = reg['p_bad'].fillna(0.5)
    reg['heuristic_value_weight'], heuristic_threshold = volatility_benchmark(reg['avg_ivol'],
        min_train_months)
    reg['heuristic_threshold'] = heuristic_threshold
    return (reg, fit_diagnostics, heuristic_threshold)


def main():
    """Run the analysis and export common-sample comparisons, diagnostics and charts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (MONTHLY_FILE, ANNUAL_FILE, FACTOR_FILE):
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}. Supply the course files in data/ or set HMM_DATA_DIR.')
    factors, reg = build_factor_inputs()
    regime, fit_diagnostics, heuristic_threshold = estimate_regimes(reg)
    min_train_months = 36

    df = factors.merge(regime[['mdate',
                'p_bad',
                'state',
                'is_warmup',
                'avg_ivol',
                'heuristic_value_weight',
                'heuristic_threshold']],
        on='mdate',
        how='inner',
        validate='one_to_one').sort_values('mdate')

    market_factors = pd.read_csv(FACTOR_FILE)
    market_factors['mdate'] = pd.to_datetime(market_factors['mdate'].astype(str), format='%Y%m')
    before = len(df)
    df = df.merge(market_factors[['mdate',
                'MktRF',
                'RF']],
        on='mdate',
        how='inner',
        validate='one_to_one')
    df = df.dropna(subset=['MOM', 'VALUE', 'p_bad', 'MktRF', 'RF'])
    df = df.sort_values('mdate').reset_index(drop=True)
    if len(df) < 24:
        raise ValueError('Fewer than 24 aligned evaluation months.')
    if not np.isfinite(df[['MOM', 'VALUE', 'p_bad', 'MktRF', 'RF']]).all().all():
        raise ValueError('Non-finite values in evaluation data.')
    months = df['mdate'].dt.year * 12 + df['mdate'].dt.month
    if not months.diff().dropna().eq(1).all():
        raise ValueError('Evaluation months are not consecutive; inspect date alignment.')
    if not df['p_bad'].between(0, 1).all():
        raise ValueError('Invalid regime probability.')

    # Value allocation weights
    weights = pd.DataFrame({'Momentum': 0.0,
            'Value': 1.0,
            'Static 50-50': 0.5,
            HEURISTIC_NAME: df['heuristic_value_weight'],
            'HMM raw': df['p_bad'],
            'HMM clip 10-90': df['p_bad'].clip(0.1,
                0.9),
            'HMM clip 20-80': df['p_bad'].clip(0.2,
                0.8),
            'HMM clip 30-70': df['p_bad'].clip(0.3,
                0.7)},
        index=df.index)
    hmm_names = [c for c in weights if c.startswith('HMM')]
    assert np.allclose(weights.loc[df['is_warmup'], hmm_names], 0.5)
    assert np.allclose(weights.loc[df['is_warmup'], HEURISTIC_NAME], 0.5)

    # Allocation-change cost proxy
    turnover = 2 * weights.diff().abs().fillna(0.0)
    cost = turnover * COST_PER_UNIT_TRADED
    gross = pd.DataFrame({name: (1 - weights[name]) * df['MOM'] + weights[name] * df['VALUE'] for name in weights})
    net = gross - cost

    # Common-sample performance comparison
    rows = []
    regression_reports = []
    for name in weights:
        stats, report = performance(gross[name], df['RF'], df['MktRF'])
        net_stats, net_report = performance(net[name], df['RF'], df['MktRF'])
        rows.append({'Strategy': name,
                'Start': df['mdate'].iloc[0].date(),
                'End': df['mdate'].iloc[-1].date(),
                'Months': len(df),
                'Warmup_months': int(df['is_warmup'].sum()),
                **stats,
                'Annual_allocation_change_proxy': 12 * turnover[name].mean(),
                'Annual_allocation_cost_proxy': 12 * cost[name].mean(),
                'Proxy_net_annual_alpha': net_stats['Annual_alpha'],
                'Proxy_net_Sharpe': net_stats['Sharpe'],
                'Proxy_net_max_drawdown': net_stats['Max_drawdown']})
        regression_reports.append(f'\n{name}: gross, factor excess returns\n{report}\n\n{name}: after allocation-cost proxy, factor excess returns\n{net_report}\n')

    # Export results
    summary = pd.DataFrame(rows).set_index('Strategy')
    summary.to_csv(OUTPUT_DIR / 'performance_summary.csv')
    (OUTPUT_DIR / 'regression_reports.txt').write_text('\n'.join(regression_reports),
        encoding='utf-8')
    pd.DataFrame(fit_diagnostics).to_csv(OUTPUT_DIR / 'hmm_fit_diagnostics.csv', index=False)
    df.to_csv(OUTPUT_DIR / 'aligned_inputs_and_regimes.csv', index=False)

    monthly_parts = []
    for name in weights:
        monthly_parts.append(pd.DataFrame({'date': df['mdate'],
                    'strategy': name,
                    'is_warmup': df['is_warmup'],
                    'p_bad': df['p_bad'],
                    'lagged_avg_ivol': df['avg_ivol'],
                    'heuristic_threshold': df['heuristic_threshold'],
                    'value_weight': weights[name],
                    'momentum_weight': 1 - weights[name],
                    'gross_factor_spread': gross[name],
                    'allocation_change_proxy': turnover[name],
                    'allocation_cost_proxy': cost[name],
                    'after_cost_factor_spread': net[name]}))
    pd.concat(monthly_parts,
        ignore_index=True).to_csv(OUTPUT_DIR / 'monthly_strategy_results.csv',
            index=False)

    correlation = df[['MOM', 'VALUE']].corr()
    correlation.to_csv(OUTPUT_DIR / 'factor_correlation.csv')

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for name in gross:
        wealth, drawdown = wealth_and_drawdown(df['RF'] + gross[name])
        axes[0].plot(df['mdate'], np.log(wealth), label=name)
        axes[1].plot(df['mdate'], 100 * drawdown, label=name)
    axes[0].set_title('RF + factor spread: full period including 50/50 warm-up')
    axes[0].set_ylabel('Log wealth index')
    axes[0].legend(ncol=2)
    axes[1].set_ylabel('Drawdown (%)')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'performance_and_drawdowns.png', dpi=160)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name in hmm_names:
        axes[0].plot(df['mdate'], weights[name], label=name, alpha=0.8)
    axes[0].plot(df['mdate'],
        weights[HEURISTIC_NAME],
        label=HEURISTIC_NAME,
        color='purple',
        linestyle='--',
        alpha=0.8)
    axes[0].axhline(0.5, color='black', linestyle='--', label='Static 50/50')
    axes[0].set_ylabel('Value allocation')
    axes[0].legend(ncol=2)
    axes[1].plot(df['mdate'], df['p_bad'], label='Estimated probability / 0.5 warm-up')
    axes[1].set_ylabel('Higher-risk state probability')
    live = df.loc[~df['is_warmup'], 'mdate']
    if not live.empty:
        for ax in axes:
            ax.axvline(live.iloc[0], color='grey', linestyle=':')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'allocations_and_regimes.png', dpi=160)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    subset = summary.loc[hmm_names]
    axes[0].bar(hmm_names, 100 * subset['Annual_alpha'], label='Gross')
    axes[0].plot(hmm_names,
        100 * subset['Proxy_net_annual_alpha'],
        'ko--',
        label='After allocation-cost proxy')
    axes[0].set_ylabel('Annualised CAPM alpha (%) â€” factor excess returns')
    axes[0].legend()
    axes[1].bar(hmm_names, 100 * subset['Annual_allocation_change_proxy'])
    axes[1].set_ylabel('Annual allocation-change proxy (%)')
    for ax in axes:
        ax.tick_params(axis='x', rotation=25)
        ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'clipping_comparison.png', dpi=160)

    # Analysis assumptions
    notes = f"ANALYSIS CONVENTIONS\nMonthly input: {MONTHLY_FILE}\nAnnual input: {ANNUAL_FILE}\nFactor input: {FACTOR_FILE}\nCommon evaluation sample: {df.mdate.min().date()} to {df.mdate.max().date()}.\nMonths removed by factor alignment / missing evaluation values: {before - len(df)}.\nHMM warm-up months retained: {int(df.is_warmup.sum())}; HMM and heuristic use 50/50.\nAll variants use the SAME HMM probabilities: clipping changes weights, not probabilities.\n\nHEURISTIC BENCHMARK\nUses lagged average stock idiosyncratic volatility, without HMM probabilities.\nThreshold is the median of the initial {min_train_months} regime-input observations,\nfrom {reg.mdate.iloc[0].date()} to {reg.mdate.iloc[min_train_months - 1].date()}.\nFrozen threshold: {heuristic_threshold:.10g}; never updated using future observations.\nAfter warm-up: above threshold = 70% VALUE / 30% MOM;\nat or below threshold = 30% VALUE / 70% MOM.\nWarm-up stays 50/50. Same evaluation dates and cost proxy as other strategies.\nThis is a simple comparison rule, not an industry-standard or optimised allocation.\n\nRETURN ACCOUNTING\nFactor spreads are excess returns: CAPM and Sharpe use the spread directly.\nWealth and drawdown compound 1 + RF + factor spread. This assumes one unit\nof capital earning RF supports one unit long and one unit short exposure.\nAfter-cost metrics subtract the allocation-cost proxy from the spread first.\nBorrowing, collateral and shorting frictions beyond this proxy are not modelled.\nAlpha is monthly intercept times 12. Standard deviations use ddof=1 consistently.\nAppraisal ratio = sqrt(12) * monthly alpha / residual standard deviation.\nCAPM inference uses {COV_TYPE}; HAC maxlags={HAC_LAGS} if selected.\nKurtosis is Pearson kurtosis (normal=3). Drawdown includes initial wealth=1.\n\nCOSTS AND TURNOVER\nProxy = abs(change in MOM weight) + abs(change in VALUE weight), from each\nvariant's actual weights; first row is zero (opening trades excluded).\nAnnualisation is 12 times the mean across all evaluation months.\nCost rate = {COST_PER_UNIT_TRADED:.4f} per unit of absolute allocation change.\nThis is an allocation-change cost sensitivity, not actual traded turnover:\nit ignores drift, stock-level reconstitution, short borrow and other costs.\nStatic weights have zero proxy; this DOES NOT mean a static factor strategy is free.\nProxy net alpha is re-estimated from monthly after-cost returns.\n\nRESEARCH LIMITATIONS\nJuly accounting-data availability is assumed, not verified.\nivol construction and original point-in-time availability remain unverified.\nAnnual value assignments retain the original carry-forward convention.\nCalendar lag correction is retained; raw state labelling score is unchanged.\nInspect HMM fit diagnostics and warnings; a successful run does not certify validity.\nClipping bounds were selected after inspecting historical results: exploratory.\nThe source now uses NL/FR/BE stocks and European factors, not the earlier OOS files.\nNo VIX network download is required. All analysis is local and reproducible from inputs.\n"
    (OUTPUT_DIR / 'analysis_notes.txt').write_text(notes, encoding='utf-8')

    columns = ['Annual_alpha',
        'Appraisal_ratio',
        'Volatility_annual',
        'Sharpe',
        'Max_drawdown',
        'Annual_allocation_change_proxy',
        'Proxy_net_annual_alpha']
    print('\nCOMPARISON (returns expressed as decimals)')
    print(summary[columns].round(4).to_string())
    print('\nFactor correlation:\n', correlation)
    print(f'\nSaved all results to: {OUTPUT_DIR.resolve()}')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close('all')
if __name__ == '__main__':
    main()
