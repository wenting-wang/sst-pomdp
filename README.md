# sst-pomdp

this is ...

## Scripts

* `filter.py`: Filters the ABCD SST behavioral dataset based on quality control criteria (e.g., performance flags, trial counts) and copies the corresponding valid raw data files into a clean destination folder.

* `stats.py`: Calculates behavioral statistics (SSR, reaction times) from raw SST files and merges them with demographic, CBCL (ADHD scores), medication, and IQ data to create a comprehensive analysis dataset.

* `metrics.py`: Contains mathematical functions to calculate behavioral statistics (e.g., SSRT, accuracy, inhibition slopes) and distance metrics (Wasserstein, Kolmogorov-Smirnov) used to quantify the discrepancy between observed human data and model simulations.

* `preprocessing.py`: Processes raw ABCD SST data files by categorizing trials into outcomes (GS, GE, GM, SS, SE), discretizing reaction times and stop-signal delays into 25ms steps, and formatting the data for model input.

* `ppc.py`: Performs Posterior Predictive Checks (PPC) by simulating the cognitive model using inferred parameters (MAP estimates) and calculating discrepancy metrics (e.g., Wasserstein distance, response proportions) between the simulated and observed behavioral data to validate model fit.

* `ppc_analysis.py`: Visualizes the results of the Posterior Predictive Checks. It loads the calculated discrepancy metrics, filters for valid baseline subjects, and generates a grid of histograms showing the distribution of model fit errors (e.g., Wasserstein distance, percentage error) across the population.

* `model_fit_analysis.py`: Selects three representative subjects (Best, Median, and Worst fit) based on the calculated PPC metrics and visualizes their model performance. It generates a detailed 5x3 plot grid comparing observed vs. simulated behavior across outcome rates, RT distributions, SSD dynamics, and inhibition functions.

