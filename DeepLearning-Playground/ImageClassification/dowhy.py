import dowhy.api
import dowhy.datasets

data = dowhy.datasets.linear_dataset(beta=5,
                                     num_common_causes=1,
                                     num_instruments=0,
                                     num_samples=1000,
                                     treatment_is_binary=True)

# data['df'] is just a regular pandas.DataFrame
data['df'].causal.do(x='v0',  # name of treatment variable
                     variable_types={'v0': 'b', 'y': 'c', 'W0': 'c'},
                     outcome='y',
                     common_causes=['W0']).groupby('v0').mean().plot(y='y', kind='bar')
