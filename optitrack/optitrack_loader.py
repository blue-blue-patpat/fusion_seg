import pandas

def csv_parser(filename):
    with open(filename, 'r') as f:
        head = f.readline()
        head_list = head.split(",")[::2]
        head_value = head.split(",")[1::2]
        head_dict = dict(zip(head_list, head_value))
        # print(head_dict)

    df = pandas.read_csv(filename, skiprows=5, usecols=range(374))
    print(df)
        # start_time = df[]

if __name__ == "__main__":
    csv_parser("/home/nesc525/chen/3DSVC/ignoredata/optitrack_files/Take 2021-08-17 09.24.57 PM.csv")