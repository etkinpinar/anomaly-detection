import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# converts time (h:m:s) to seconds
def time_to_sec(time):

    time = time.split(':')
    sec = int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])
    return sec


def parse_logs(path):

    session_time = 1500  # minimum time past to split two sessions
    session_id = 0
    columns = ["date", "time", "s-ip", "cs-method", "cs-uri-stem", "cs-uri-query", "s-port", "cs-username", "c-ip",
               "cs(User-Agent)", "cs(Referer)", "sc-status", "sc-substatus", "sc-win32-status", "time-taken"]

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            print("parsing", name, "...")
            data = pd.read_csv(os.path.join(root, name), sep=' ', skiprows=4, names=columns
                               ).sort_values(by=["c-ip", "time"]).reset_index(drop=True)

            session_id += 1
            data["page-time"] = 0
            data["session-id"] = session_id

            sec_old = time_to_sec(data.iloc[0]['time'])
            ip_old = data.iloc[0]["c-ip"]

            for i in range(1, len(data)):

                col = data.iloc[i]
                sec_new = time_to_sec(col['time'])
                ip_new = col['c-ip']

                page_time = sec_new - sec_old

                if ip_new != ip_old or page_time > session_time:
                    session_id += 1
                    # data.at[i - 1, "page-time"] = 0
                else:
                    data.at[i - 1, "page-time"] = page_time

                sec_old = sec_new
                ip_old = ip_new
                data.at[i, "session-id"] = session_id

            data = data.groupby('session-id').agg({'session-id': ['first'],
                                                   'c-ip': ['first', 'count'],
                                                   'time-taken': ['mean', 'sum'],
                                                   'cs-uri-stem': ['nunique'],
                                                   'page-time': ['mean', 'sum']}).reset_index(drop=True)

            data.columns = data.columns.droplevel(0)
            data.columns = ["session-id", "user-ip", "clicks", "avg-time-taken", "total-time-taken", "unique-pages",
                            "avg-time-per-page", "session-time"]

            data.to_csv("dataframes/" + name + ".csv")


def merge_sessions(path):

    data_to_merge = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            print("merging", name, "...")
            data_to_merge.append(pd.read_csv(os.path.join(root, name)))

    data = pd.concat(data_to_merge, ignore_index=True)

    # to drop the first Unnamed:0 colon
    data.drop(data.filter(regex="Unnamed"), axis=1, inplace=True)
    data.to_csv("data.csv", index=False)


def box_plot():
    data = pd.read_csv("data.csv")

    box_columns = ['clicks', 'avg-time-taken', 'total-time-taken', 'avg-time-per-page', 'session-time']

    col_min = []
    col_max = []

    for i, cols in enumerate(box_columns):
        plot = plt.boxplot(data=data, x=cols)

        col_min.append(plot['whiskers'][0].get_ydata()[1])
        col_max.append(plot['whiskers'][1].get_ydata()[1])

    data['anomaly'] = 0

    for i in range(len(data)):
        data_row = data.iloc[i]
        anomaly_cols = 0
        for j, cols in enumerate(box_columns):
            if data_row[cols] > col_max[j] or data_row[cols] < col_min[j]:
                anomaly_cols += 1

        if anomaly_cols > 3:
            data.at[i, "anomaly"] = 1

    data.to_csv("data-anomaly.csv", index=False)


# parse_logs("logs/")
# merge_sessions("dataframes")
# box_plot()

dat = pd.read_csv("data-anomaly.csv")
print(dat["anomaly"].sum())





