from blinker import Blinker

b = Blinker(frame_interval_s=0.001, min_snr=5.0)  # or Blinker(AnalysisParams(...))
b.load("/Users/jprice/Downloads/ss_stack_0.tiff").analyze()

print(b.summary())  # key metrics as text
on, off = b.dwell_times()  # calculate ON/OFF times (pooled, seconds)
b.plot_summary()  # ← the summary output
b.plot_overview()
b.plot_statistics()
b.plot_trace(0)
b.save("results")  # tables + JSON + figures incl. summary.png
