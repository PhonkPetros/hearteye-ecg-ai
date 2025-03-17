import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Plots:
    def __init__(self, labeled_data=None, unlabeled_data=None, mixed_data=None, sample_rate=None):
        """
        A class providing methods to visualize ECG datasets (labeled, unlabeled, and mixed),
        including raw ECG signals, feature distributions, missing data patterns, QRST comparisons,
        and demographic comparisons.

        Parameters
        ----------
        labeled_data : pandas.DataFrame or None
            DataFrame for the labeled ECG dataset.
        unlabeled_data : pandas.DataFrame or None
            DataFrame for the unlabeled ECG dataset.
        mixed_data : pandas.DataFrame or None
            DataFrame for a combined or mixed ECG dataset.
        sample_rate : float or None
            Sampling rate (Hz) for ECG signals, if available, to display x-axis in seconds.
        """
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data
        self.mixed_data = mixed_data
        self.sample_rate = sample_rate

    def _select_dataset(self, dataset_name):
        """
        Internal helper to pick the appropriate dataset based on the provided string.
        Returns the corresponding DataFrame or None if invalid.
        """
        dataset_name = dataset_name.lower()
        if dataset_name == 'labeled':
            return self.labeled_data
        elif dataset_name == 'unlabeled':
            return self.unlabeled_data
        elif dataset_name == 'mixed':
            return self.mixed_data
        else:
            raise ValueError("dataset must be one of: 'labeled', 'unlabeled', 'mixed'")

    def plot_ecg_signal(self, dataset='labeled', record_index=0, lead=None,
                        gender=None, age_range=None, diagnosis=None, save=False):
        """
        Plot an ECG waveform from the specified dataset on a single figure.
        You can filter by gender, age range, or diagnosis if those columns exist.

        Parameters
        ----------
        dataset : str
            Which dataset to plot from: 'labeled', 'unlabeled', or 'mixed'.
        record_index : int
            Index (row) of the record in the dataset to plot.
        lead : str or None
            If the signal column contains multiple leads in a dict-like or multi-column format,
            specify which lead to plot. Otherwise, the first numeric data found is used.
        gender : str or None
            If provided, filter by 'gender' == this value (e.g., "M" or "F").
        age_range : tuple or list of two values, optional
            If provided, filter to only rows where age is in [age_min, age_max].
        diagnosis : str or list of str or None
            If provided, filter by diagnosis in the 'diagnosis' column.
        save : bool
            If True, save the figure as a PNG file in the 'analysis' folder.
        """
        df = self._select_dataset(dataset)
        if df is None or df.empty:
            print(f"No data available in '{dataset}' dataset.")
            return

        df_filt = df.copy()

        # Apply optional filters
        if gender and 'gender' in df_filt.columns:
            df_filt = df_filt[df_filt['gender'] == gender]
        if age_range and 'age' in df_filt.columns:
            df_filt = df_filt[(df_filt['age'] >= age_range[0]) & (df_filt['age'] <= age_range[1])]
        if diagnosis and 'diagnosis' in df_filt.columns:
            if isinstance(diagnosis, str):
                diagnosis = [diagnosis]
            df_filt = df_filt[df_filt['diagnosis'].isin(diagnosis)]

        if df_filt.empty:
            print("No data matches the specified filters.")
            return

        # Ensure record_index is valid after filtering
        if record_index >= len(df_filt):
            print(f"record_index {record_index} is out of range after filtering. Max index: {len(df_filt)-1}")
            return

        # Get the row we want
        row = df_filt.iloc[record_index]

        # Attempt to find the ECG signal
        signal_data = None

        # Case A: the row has a single column "signal" that stores the waveform (list, array, dict, etc.)
        if "signal" in df_filt.columns:
            signal_data = row["signal"]
            # If we have multiple leads in a dict-like structure, pick the requested lead
            if lead and isinstance(signal_data, (dict, pd.Series)):
                if lead in signal_data:
                    signal_data = signal_data[lead]
                else:
                    print(f"Lead '{lead}' not found in signal data keys: {list(signal_data.keys())}")
                    return
        else:
            # Case B: multiple columns, each for a different lead or dimension
            # find a numeric column or the specified 'lead' column
            numeric_cols = df_filt.select_dtypes(include=[np.number]).columns
            if lead and lead in numeric_cols:
                signal_data = row[lead]
            elif len(numeric_cols) > 0:
                # fallback: first numeric column
                signal_data = row[numeric_cols[0]]
            else:
                print("No numeric signal columns found.")
                return

        # Convert to a numpy array
        if isinstance(signal_data, (list, tuple, np.ndarray, pd.Series)):
            signal = np.array(signal_data, dtype=float)
        else:
            print("ECG signal could not be interpreted as an array.")
            return

        # Drop NaNs from the signal
        mask = ~np.isnan(signal)
        signal = signal[mask]

        # Time axis
        if self.sample_rate:
            time_axis = np.arange(len(signal)) / self.sample_rate
            x_label = "Time (s)"
        else:
            time_axis = np.arange(len(signal))
            x_label = "Sample Index"

        plt.figure()
        plt.plot(time_axis, signal)
        plt.title(f"ECG Waveform - {dataset.capitalize()} (Record {record_index})")
        plt.xlabel(x_label)
        plt.ylabel("Amplitude")
        plt.tight_layout()

        if save:
            os.makedirs('analysis', exist_ok=True)
            f_name = f"{dataset}_ecg_record{record_index}.png"
            plt.savefig(os.path.join('analysis', f_name))
        else:
            plt.show()

    def plot_feature_distribution(self, feature, dataset='labeled', bins=30,
                                  gender=None, age_range=None, diagnosis=None,
                                  save=False):
        """
        Plot a histogram of the specified feature from one dataset. Optionally filter
        by gender, age range, and diagnosis. Missing values are dropped.

        Parameters
        ----------
        feature : str
            Name of the feature column to plot (e.g., 'QRS_duration').
        dataset : str
            'labeled', 'unlabeled', or 'mixed'.
        bins : int
            Number of histogram bins.
        gender : str or None
            Filter by gender if 'gender' column exists.
        age_range : (min_age, max_age) or None
            Filter by age range if 'age' column exists.
        diagnosis : str or list of str or None
            Filter by diagnosis values if 'diagnosis' column exists.
        save : bool
            If True, saves figure to 'analysis'.
        """
        df = self._select_dataset(dataset)
        if df is None:
            print(f"No data in '{dataset}' dataset.")
            return
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in '{dataset}' dataset.")
            return

        df_filt = df.copy()
        # Filters
        if gender and 'gender' in df_filt.columns:
            df_filt = df_filt[df_filt['gender'] == gender]
        if age_range and 'age' in df_filt.columns:
            df_filt = df_filt[(df_filt['age'] >= age_range[0]) & (df_filt['age'] <= age_range[1])]
        if diagnosis and 'diagnosis' in df_filt.columns:
            if isinstance(diagnosis, str):
                diagnosis = [diagnosis]
            df_filt = df_filt[df_filt['diagnosis'].isin(diagnosis)]

        if df_filt.empty:
            print("No data matches the specified filters for distribution plot.")
            return

        data_to_plot = df_filt[feature].dropna()
        if data_to_plot.empty:
            print(f"No valid values for feature '{feature}' after dropping NaNs.")
            return

        plt.figure()
        plt.hist(data_to_plot, bins=bins)
        plt.title(f"Distribution of {feature} - {dataset.capitalize()}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.tight_layout()

        if save:
            os.makedirs('analysis', exist_ok=True)
            f_name = f"{dataset}_{feature}_distribution.png"
            plt.savefig(os.path.join('analysis', f_name))
        else:
            plt.show()

    def plot_missing_heatmap(self, dataset='mixed', save=False):
        """
        Visualize the missing data pattern as a 2D image for the chosen dataset.
        Each cell is 1 if the value is missing, 0 if present.

        Parameters
        ----------
        dataset : str
            'labeled', 'unlabeled', 'mixed'
        save : bool
            If True, saves figure to 'analysis'.
        """
        df = self._select_dataset(dataset)
        if df is None or df.empty:
            print(f"No data available in '{dataset}' dataset.")
            return

        mask = df.isna().astype(int)  # 1=missing, 0=present
        plt.figure()
        plt.imshow(mask, aspect='auto', interpolation='none')
        plt.title(f"Missing Data Heatmap - {dataset.capitalize()}")
        plt.xlabel("Features")
        plt.ylabel("Records")
        # We won't add feature names or record IDs for large sets, but you can adapt if needed.
        plt.colorbar(label='Missing (1) / Present (0)')
        plt.tight_layout()

        if save:
            os.makedirs('analysis', exist_ok=True)
            plt.savefig(os.path.join('analysis', f"{dataset}_missing_heatmap.png"))
        else:
            plt.show()

    def plot_missing_bar(self, dataset='mixed', save=False):
        """
        Plot a bar chart of the count of missing values per column in the dataset.

        Parameters
        ----------
        dataset : str
            'labeled', 'unlabeled', 'mixed'
        save : bool
            If True, saves figure to 'analysis'.
        """
        df = self._select_dataset(dataset)
        if df is None or df.empty:
            print(f"No data available in '{dataset}' dataset.")
            return

        missing_counts = df.isna().sum()
        plt.figure()
        plt.bar(x=missing_counts.index, height=missing_counts.values)
        plt.title(f"Missing Values per Feature - {dataset.capitalize()}")
        plt.xlabel("Features")
        plt.ylabel("Count of Missing Values")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save:
            os.makedirs('analysis', exist_ok=True)
            plt.savefig(os.path.join('analysis', f"{dataset}_missing_bar.png"))
        else:
            plt.show()

    def plot_qrst_analysis(self, features=('QRS_duration', 'QT_interval'), save=False):
        """
        Compare one or more ECG features (defaults to QRS_duration & QT_interval) across
        all available datasets using boxplots.

        Parameters
        ----------
        features : tuple or list of str
            Feature columns to compare across labeled, unlabeled, mixed datasets.
        save : bool
            If True, saves figure to 'analysis'.
        """
        available_data = []
        labels = []
        # We combine each dataset's features if they exist
        for label, df in [('Labeled', self.labeled_data),
                          ('Unlabeled', self.unlabeled_data),
                          ('Mixed', self.mixed_data)]:
            if df is not None and not df.empty:
                available_data.append((label, df))
        
        # If no datasets are available, exit
        if not available_data:
            print("No datasets available for QRST analysis.")
            return

        # We'll handle multiple features by creating a group of boxplots side by side for each dataset
        # For a single feature, each dataset is a separate box; for multiple features, we create grouped boxes.

        # Collect data in a structure that can be used for a manual boxplot grouping.
        # E.g., if features = ('QRS_duration','QT_interval'), for each dataset we gather their data in a list.
        data_for_plot = []
        dataset_names = []
        feature_names = []

        # We'll store data in shape: data_for_plot[f] = [ values_for_labeled, values_for_unlabeled, ... ]
        # for each feature f. Then we can boxplot them side by side.
        for f in features:
            data_for_plot.append([])
            feature_names.append(f)

        used_dataset_labels = []
        for (label, df) in available_data:
            used_dataset_labels.append(label)
            for idx, f in enumerate(features):
                if f in df.columns:
                    col_data = df[f].dropna().values
                else:
                    col_data = np.array([])  # no data
                data_for_plot[idx].append(col_data)

        # Now we have data_for_plot = [ [arr_labeled, arr_unlabeled, arr_mixed],  # for feature1
        #                               [arr_labeled, arr_unlabeled, arr_mixed],  # for feature2
        #                             ]
        # Some features might not exist in some data => those arrays might be empty.

        # We'll create one figure. If there are multiple features, we group them in the x-axis manually.
        # We'll do a manual approach: each feature is a group, and within that group we have multiple boxes
        # (one per dataset).
        
        plt.figure()
        positions = []
        data_values = []
        # We'll keep track of the x-position offset
        current_pos = 1
        gap = 1.5  # gap between different features

        # We'll build a custom set of boxplots by repeatedly calling plt.boxplot for each feature
        for feature_idx, feature_data_list in enumerate(data_for_plot):
            # feature_data_list is e.g. [arr_labeled, arr_unlabeled, arr_mixed]
            # Filter out empty lists if any, but we must preserve ordering
            # We'll plot them only if there's at least some non-empty data
            if all(len(a) == 0 for a in feature_data_list):
                # No data for this feature in any dataset
                continue

            # We'll define positions for each dataset's box in this group
            n_datasets = len(feature_data_list)
            # We'll place them at consecutive x-locations: e.g. current_pos, current_pos+1, ...
            dataset_positions = list(range(current_pos, current_pos + n_datasets))
            # Add the data
            plot_data = [arr for arr in feature_data_list]

            plt.boxplot(plot_data, positions=dataset_positions, widths=0.6)

            # We'll place a label for the midpoint of these boxes on the x-axis = mean of these positions
            mid_x = np.mean(dataset_positions)
            plt.text(mid_x, plt.ylim()[1]*1.01, feature_names[feature_idx],
                     ha='center', va='bottom')

            current_pos += n_datasets + gap

        plt.title("QRST Feature Comparison Across Datasets")
        plt.xlabel("Features (grouped)")
        plt.ylabel("Value")
        # We'll create a custom x-axis or let them remain numeric.
        # If we want a legend for dataset, we can't directly do it in boxplot easily,
        # but we can add text or disclaimers. We'll note that the order is the same as used_dataset_labels.
        # We'll place them near the top.
        label_str = f"Datasets order in each group: {', '.join(used_dataset_labels)}"
        plt.text(0.5, 1.02, label_str, ha='center', va='bottom', transform=plt.gca().transAxes)

        plt.tight_layout()
        if save:
            os.makedirs('analysis', exist_ok=True)
            plt.savefig(os.path.join('analysis', "qrst_comparison.png"))
        else:
            plt.show()

    def plot_demographic_comparison(self, feature, dataset='labeled', by='gender',
                                    bins=30, save=False):
        """
        Compare the distribution of a feature by gender or age groups within one dataset.

        Parameters
        ----------
        feature : str
            Column name of the feature to compare.
        dataset : str
            Which dataset to use: 'labeled', 'unlabeled', 'mixed'.
        by : str
            'gender' -> Overlays two histograms for M/F if data available.
            'age' -> Groups into bins of age ranges and shows boxplots.
        bins : int
            Number of bins for hist if by='gender'.
        save : bool
            If True, saves figure to 'analysis'.
        """
        df = self._select_dataset(dataset)
        if df is None or df.empty:
            print(f"No data in '{dataset}' dataset.")
            return
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in {dataset} dataset.")
            return

        df_filt = df.dropna(subset=[feature])

        plt.figure()
        if by == 'gender':
            if 'gender' not in df_filt.columns:
                print("No 'gender' column in dataset.")
                return
            male_data = df_filt[df_filt['gender'] == 'Male'][feature].dropna()
            female_data = df_filt[df_filt['gender'] == 'Female'][feature].dropna()

            if len(male_data) == 0 and len(female_data) == 0:
                print("No gender data available to compare.")
                return

            # Plot male histogram
            plt.hist(male_data, bins=bins, alpha=0.5, label='Male', density=True)
            # Overplot female histogram
            plt.hist(female_data, bins=bins, alpha=0.5, label='Female', density=True)

            plt.title(f"{feature} Distribution by Gender - {dataset.capitalize()}")
            plt.xlabel(feature)
            plt.ylabel("Density")
            plt.legend()

        elif by == 'age':
            if 'age' not in df_filt.columns:
                print("No 'age' column in dataset.")
                return
            # Bin into 5 groups: [0-20), [20-40), [40-60), [60-80), [80+)
            bins_edges = [0, 20, 40, 60, 80, 200]
            labels = ["0-19", "20-39", "40-59", "60-79", "80+"]

            df_filt['age_group'] = pd.cut(df_filt['age'], bins=bins_edges, labels=labels, right=False)
            # We'll create a list of arrays, one per age group, for boxplot
            grouped = df_filt.groupby('age_group')[feature]

            # Prepare data in the correct order of bins
            data_boxplot = []
            x_labels = []
            for grp_label in labels:
                grp_data = grouped.get_group(grp_label) if grp_label in grouped.groups else pd.Series([])
                data_boxplot.append(grp_data.dropna().values)
                x_labels.append(grp_label)

            plt.boxplot(data_boxplot, positions=range(1, len(data_boxplot)+1))
            plt.title(f"{feature} by Age Group - {dataset.capitalize()}")
            plt.xlabel("Age Group")
            plt.ylabel(feature)
            # Tweak x-ticks
            plt.xticks(range(1, len(data_boxplot)+1), x_labels, rotation=0)
        else:
            print("Argument 'by' must be 'gender' or 'age'.")
            return

        plt.tight_layout()

        if save:
            os.makedirs('analysis', exist_ok=True)
            fname = f"{dataset}_{feature}_by_{by}.png"
            plt.savefig(os.path.join('analysis', fname))
        else:
            plt.show()

    def save_all_plots(self):
        """
        Example convenience method that calls various plotting functions with save=True
        to generate a batch of plots for each dataset in the 'analysis' folder.
        Adjust or extend as needed for your project.
        """
        os.makedirs('analysis', exist_ok=True)

        # 1) ECG signal example
        if self.labeled_data is not None and not self.labeled_data.empty:
            self.plot_ecg_signal(dataset='labeled', record_index=0, save=True)
        if self.unlabeled_data is not None and not self.unlabeled_data.empty:
            self.plot_ecg_signal(dataset='unlabeled', record_index=0, save=True)
        if self.mixed_data is not None and not self.mixed_data.empty:
            self.plot_ecg_signal(dataset='mixed', record_index=0, save=True)

        # 2) Feature distribution example (QRS_duration if it exists)
        for name in ['labeled','unlabeled','mixed']:
            df = self._select_dataset(name)
            if df is not None and not df.empty and 'QRS_duration' in df.columns:
                self.plot_feature_distribution(feature='QRS_duration', dataset=name, save=True)

        # 3) Missing data visualizations
        for name in ['labeled','unlabeled','mixed']:
            df = self._select_dataset(name)
            if df is not None and not df.empty:
                self.plot_missing_heatmap(dataset=name, save=True)
                self.plot_missing_bar(dataset=name, save=True)

        # 4) QRST comparison if multiple datasets exist
        used = sum([self.labeled_data is not None, self.unlabeled_data is not None, self.mixed_data is not None])
        if used > 1:
            self.plot_qrst_analysis(save=True)

        # 5) Demographic comparisons in labeled or mixed (often unlabeled lacks demo info)
        for name in ['labeled','mixed']:
            df = self._select_dataset(name)
            if df is not None and not df.empty and 'QRS_duration' in df.columns:
                # Compare QRS_duration by gender if 'gender' in columns
                if 'gender' in df.columns:
                    self.plot_demographic_comparison(feature='QRS_duration', dataset=name, by='gender', save=True)
                # Compare QRS_duration by age if 'age' in columns
                if 'age' in df.columns:
                    self.plot_demographic_comparison(feature='QRS_duration', dataset=name, by='age', save=True)


def plot_arrhythmia_comparison(self, feature='QRS_duration', dataset='labeled', bins=30, save=False):
    """
    Compare the distribution of a numeric feature (e.g., QRS_duration) between
    arrhythmia vs. non-arrhythmia records in the specified dataset.

    Parameters
    ----------
    feature : str
        Name of the numeric feature column to plot.
    dataset : str
        Which dataset to draw from: 'labeled', 'unlabeled', or 'mixed'.
    bins : int
        Number of bins for the histogram.
    save : bool
        If True, saves the figure in the 'analysis' folder; otherwise shows it.
    """
    df = self._select_dataset(dataset)
    if df is None or df.empty:
        print(f"No data in '{dataset}' dataset.")
        return

    # We need an 'arrhythmia' column with True/False (or 1/0). If missing, exit.
    if 'arrhythmia' not in df.columns:
        print("No 'arrhythmia' column found in this dataset.")
        return

    # Check that the feature exists
    if feature not in df.columns:
        print(f"Feature '{feature}' not found in {dataset} dataset.")
        return

    # Drop NaNs for the feature of interest
    df_filt = df.dropna(subset=[feature])

    # Separate into arrhythmic vs. non-arrhythmic
    arr_df = df_filt[df_filt['arrhythmia'] == True]
    non_arr_df = df_filt[df_filt['arrhythmia'] == False]

    if arr_df.empty and non_arr_df.empty:
        print("No arrhythmia or non-arrhythmia data available to compare.")
        return

    # Extract numeric arrays
    arr_data = arr_df[feature].values
    non_arr_data = non_arr_df[feature].values

    # Create a figure and plot two overlaid histograms
    plt.figure()
    plt.hist(arr_data, bins=bins, alpha=0.5, label='Arrhythmia', density=True)
    plt.hist(non_arr_data, bins=bins, alpha=0.5, label='Non-Arrhythmia', density=True)

    plt.title(f"Arrhythmia vs. Non-Arrhythmia: {feature} - {dataset.capitalize()}")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    if save:
        os.makedirs('analysis', exist_ok=True)
        fname = f"{dataset}_{feature}_arrhythmia_comparison.png"
        plt.savefig(os.path.join('analysis', fname))
    else:
        plt.show()


def plot_abnormal_heart_rate_comparison(self, hr_column='heart_rate', dataset='labeled', bins=30, save=False):
    """
    Compare the distribution of heart rate (or any numeric column) between
    abnormal vs. normal heart_rate records in the specified dataset.

    Parameters
    ----------
    hr_column : str
        Name of the numeric column representing heart rate (e.g., 'heart_rate').
    dataset : str
        Which dataset to draw from: 'labeled', 'unlabeled', or 'mixed'.
    bins : int
        Number of bins for the histogram.
    save : bool
        If True, saves the figure in the 'analysis' folder; otherwise shows it.
    """
    df = self._select_dataset(dataset)
    if df is None or df.empty:
        print(f"No data in '{dataset}' dataset.")
        return

    # Check that the needed columns exist
    if hr_column not in df.columns:
        print(f"'{hr_column}' not found in {dataset} dataset.")
        return

    if 'abnormal_hr' not in df.columns:
        print("No 'abnormal_hr' column found in this dataset. Please add a boolean column indicating abnormal HR.")
        return

    # Drop NaNs for the heart-rate column
    df_filt = df.dropna(subset=[hr_column])

    # Separate into abnormal vs. normal
    abnormal_df = df_filt[df_filt['abnormal_hr'] == True]
    normal_df = df_filt[df_filt['abnormal_hr'] == False]

    if abnormal_df.empty and normal_df.empty:
        print("No abnormal_hr or normal_hr data available to compare.")
        return

    # Extract numeric arrays
    abn_data = abnormal_df[hr_column].values
    nor_data = normal_df[hr_column].values

    # Create a figure and plot two overlaid histograms
    plt.figure()
    plt.hist(abn_data, bins=bins, alpha=0.5, label='Abnormal HR', density=True)
    plt.hist(nor_data, bins=bins, alpha=0.5, label='Normal HR', density=True)

    plt.title(f"Abnormal vs. Normal Heart Rate - {dataset.capitalize()}")
    plt.xlabel(hr_column)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    if save:
        os.makedirs('analysis', exist_ok=True)
        fname = f"{dataset}_{hr_column}_abnormal_hr_comparison.png"
        plt.savefig(os.path.join('analysis', fname))
    else:
        plt.show()

