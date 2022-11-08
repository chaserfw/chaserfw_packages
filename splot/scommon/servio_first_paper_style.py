print ('[INFO]: Saving using matplotlib')
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

matplotlib.use('pdf')
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=False)
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')
plt.rc('axes', labelsize='medium')


plt.title(args.title)
plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)

for i, leaks in enumerate(leakages):
	plt.grid(True)
	plt.plot(time, leaks, linestyle='-' if i % 2 == 0 else 'dotted', label='Trace_' + str(i) if len(args.legends) == 0 else args.legends[i], linewidth=0.1)
	#plt.legend()
plt.savefig(os.path.join(os.getcwd(), '{}.pdf'.format(args.plotname)))