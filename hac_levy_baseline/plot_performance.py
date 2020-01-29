import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
import pickle
p_log = pickle.load(open("hac_models/performance_log.p", "rb"))
# fig = plt.figure()
# plt.plot(p_log)
print("Performance:")
print(p_log)



