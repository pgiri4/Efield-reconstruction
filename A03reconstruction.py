from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.WARNING)
import numpy as np
from scipy import signal
import argparse
from datetime import datetime
import pickle
import os

from NuRadioReco.utilities import units
import NuRadioReco.detector.detector as detector
import NuRadioReco.modules.io.eventReader
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import electricFieldParameters as efp

import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.correlationDirectionFitter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import NuRadioReco.modules.voltageToEfieldConverterPerChannel
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.efieldTimeDirectionFitter
import NuRadioReco.modules.channelTimeWindow
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.voltageToEfieldConverter
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.utilities import medium
from radiotools import helper as hp
plt.switch_backend('agg')

plot = 1

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()voltageToEfieldConverterPerChannel = NuRadioReco.modules.voltageToEfieldConverterPerChannel.voltageToEfieldConverterPerChannel()
electricFieldSignalReconstructor = NuRadioReco.modules.electricFieldSignalReconstructor.electricFieldSignalReconstructor()
efieldTimeDirectionFitter = NuRadioReco.modules.efieldTimeDirectionFitter.efieldTimeDirectionFitter()
channelTimeWindow = NuRadioReco.modules.channelTimeWindow.channelTimeWindow()
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()

efieldTimeDirectionFitter.begin(debug=plot)
channelTimeWindow.begin(debug=False)
correlationDirectionFitter.begin(debug=False, log_level=logging.DEBUG)

electricFieldSignalReconstructor.begin(log_level=logging.WARNING)

# Parse eventfile as argument
parser = argparse.ArgumentParser(description='NuRadioSim file')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC simulation result')
args = parser.parse_args()

det = detector.Detector()

# initialize modules
eventReader = NuRadioReco.modules.io.eventReader.eventReader()
eventReader.begin(args.inputfilename)

for evt in eventReader.run():
    for station in evt.get_stations():
        station_id = station.get_id()
        t = station.get_station_time()
        det.update(t)
        channelResampler.run(evt, station, det, 50 * units.GHz)
        channelBandPassFilter.run(evt, station, det, passband=[120 * units.MHz, 300 * units.MHz], filter_type='butterabs', order=10)
        channelBandPassFilter.run(evt, station, det, passband=[10 * units.MHz, 1000 * units.MHz], filter_type='rectangular')
        correlationDirectionFitter.run(evt, station, det, n_index=1.353, ZenLim=[90 * units.deg, 180 * units.deg],
                                       AziLim=[300 * units.deg, 330 * units.deg],
                                       channel_pairs=((0, 2), (1, 3)))
        correlationDirectionFitter.run(evt, station, det, n_index=1.353, ZenLim=[90 * units.deg, 180 * units.deg],
                                       AziLim=[300 * units.deg, 330 * units.deg],
                                       channel_pairs=((6, 4), (5, 7)))
                voltageToEfieldConverterPerChannel.run(evt, station, det, pol=0)
        electricFieldSignalReconstructor.run(evt, station, det)
        if plot:
            fig, ax = plt.subplots(4, 2, sharex=True, sharey=True)
            ax = ax.flatten(order='F')
            for efield in station.get_electric_fields():
                eid = efield.get_channel_ids()[0]
                print("reconstructing Efield")
                if not eid in range(8):
                    continue
                tt = efield.get_times()  # + efield.get_trace_start_time()
                t0 = tt[0]
                tt -= t0
                ax[eid].plot(tt / units.ns, efield.get_trace()[1] / units.mV, lw=1, label='rec')
                ax[eid].plot(tt / units.ns, np.abs(signal.hilbert(efield.get_trace()[1])) / units.mV, '--', lw=1)
                ax[eid].axvline(efield[efp.signal_time] - t0, linestyle='--', lw=1)
                ax[eid].set_xlim(10, 150)
            # add simulated efields
            for efield in station.get_sim_station().get_electric_fields():
                eid = efield.get_channel_ids()[0]
            for i in range(8):
                ax[i].legend(fontsize='xx-small')
            ax[3].set_xlabel("time [ns]")
            ax[7].set_xlabel("time [ns]")
            fig.suptitle("reconstructed electric field")
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            if(not os.path.exists("plots/efields")):
                os.makedirs("plots/efields")
            fig.savefig("plots/efields/Efield.png")
            plt.close("all")

