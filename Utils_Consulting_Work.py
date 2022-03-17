import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import glob
from sklearn.preprocessing import StandardScaler


class transform_data:
    def __init__(self, data, standardscaler_cols=None, categorical_cols=None, log_cols=None):
        self.df = data
        self.continuous_cols = standardscaler_cols
        self.categorical_cols = categorical_cols
        self.log_cols = log_cols

    def log_transform_cols(self):
        self.df[self.log_cols] = self.df[self.log_cols].apply(
            lambda x: np.log(x))

    def standardize_cols(self):

        scaler = StandardScaler().fit(self.df[self.continuous_cols])
        standardized_cols = pd.DataFrame(
            scaler.transform(self.df[self.continuous_cols]))

        self.df[self.continuous_cols] = standardized_cols

    # def get_dummies_cols(self):
    #     self.df = pd.DataFrame(pd.get_dummies(
    #         self.df, columns=self.categorical_cols))

    def transform(self):
        if self.continuous_cols:
            self.standardize_cols()

        # if self.categorical_cols:
        #     self.get_dummies_cols()

        if self.log_cols:
            self.log_transform_cols()

        return self.df


class consulting_summary_stats:
    def __init__(self, data, cont_vars, cat_vars):
        self.data = data
        self.cont_vars = cont_vars
        self.cat_vars = cat_vars

    def create_results(self):
        if not os.path.exists(f'./results'):
            os.mkdir(f'./results')

    def create_folder(self, output_file):

        if not os.path.exists(f'./results/{output_file}'):
            os.mkdir(f'./results/{output_file}')

    def plot_corr_matrix(self):
        corr = self.data[self.cont_vars].corr()
        mask = np.triu(np.ones_like(
            self.data[self.cont_vars].corr(), dtype=np.bool))

        ax1 = sns.heatmap(corr, cbar=10, linewidths=2, vmax=1,
                          vmin=-1, square=True, cmap='Blues', annot=True, mask=mask)
        ax1.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
        plt.show()

        self.create_folder('heat_map')
        fig = ax1.get_figure()
        fig.savefig('./results/heat_map/heat_map_plot.png')

    def summary_stats(self):

        # Summary Statistics: Continuous
        self.create_folder('tables')
        self.data[self.cont_vars].describe().T.iloc[:, :3].round(
            2).to_csv('./results/tables/summary_stats_cont.csv')

        # Summary Statistics: Categorical
        cat_stats = []

        for col in self.cat_vars:
            temp_count = self.data[self.cat_vars][col].value_counts()
            temp_pct = self.data[self.cat_vars][col].value_counts(normalize=True)
            temp_df = pd.concat([temp_count, temp_pct], axis=1).reset_index()
            temp_df.columns = ['Variable', 'Count', 'Percent']

            cat_stats.append(temp_df)

        pd.concat(cat_stats).to_csv('./results/tables/summary_stats_cat.csv')

    def plot_cont(self):
        n_rows = int(np.ceil(len(self.cont_vars) / 2))

        plt.subplots(nrows=n_rows, ncols=2, figsize=(15, 10))
        plt.tight_layout(pad=5.0)

        for i, j in enumerate(self.cont_vars):
            plt.subplot(n_rows, 2, i + 1)
            sns.histplot(self.data[j])

        self.create_folder('plots')
        plt.savefig('./results/plots/full_figure.png')


class univariate_analysis(consulting_summary_stats):
    def __init__(self, data, cont_vars, cat_vars, outcome_vars):
        self.outcome = outcome_vars
        super().__init__(data, cont_vars, cat_vars)

    def rslts_to_csv(self, results, outcome, var):
        rslts_csv_str = results.summary().tables[1].as_csv()

        self.create_folder('univar_tables')
        self.create_folder(f'univar_tables/{outcome}')

        f = open(
            f'./results/univar_tables/{outcome}/univariate_{outcome}_{var}.csv', 'w')
        f.write(rslts_csv_str)
        f.close()

    def univar_reg(self, outcome):
        reg_vars = [i for i in self.cat_vars +
                    self.cont_vars if i not in self.outcome]

        for var in reg_vars:
            if var in self.cat_vars:
                formula = f"{outcome} ~ C({var})"
            else:
                formula = f"{outcome} ~ {var}"

            mod = smf.ols(formula, data=self.data)
            rslt = mod.fit()

            self.rslts_to_csv(rslt, outcome, var)

    def create_univar_table(self, outcome):
        # in the folder
        path = f'./results/univar_tables/{outcome}/'
        csv_files = glob.glob(os.path.join(path, "*.csv"))

        df_list = []
        for f in csv_files:

            df_univariate = pd.read_csv(f)
            df_univariate = df_univariate.iloc[1:, :].reset_index(drop=True)

            df_univariate.columns = df_univariate.columns.str.strip()
            df_univariate.columns = ['variable'] + \
                df_univariate.columns[1:].to_list()

            # df_univariate.columns[0] = 'Variable'
            df_list.append(df_univariate)

            self.create_folder('univariate_table_final')

            pd.concat(df_list).to_csv(
                f'./results/univariate_table_final/univariate_{outcome}_final_table.csv')

    def create_univariate_table(self):
        for outcome_var in self.outcome:
            self.univar_reg(outcome_var)
            self.create_univar_table(outcome_var)


def write_results_table(model, outcome):
    rslts_csv_str = model.summary().tables[1].as_csv()

    if not os.path.exists(f'./results/multivar_tables'):
        os.mkdir(f'./results/multivar_tables')

    f = open(f'./results/multivar_tables/multivar_{outcome}.csv', 'w')
    f.write(rslts_csv_str)
    f.close()
