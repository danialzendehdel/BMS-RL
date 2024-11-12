from Utilities.read_data import read_data
from Environments.Environment_BMS import BMSEnvironment


env = BMSEnvironment()

# Reset the environment to get the initial observation and info
obs, info = env.reset()

PV_data, consumption_data = read_data(path="Data/Data_PV and consumptions.xlsx")
