E. SIGNAL QUALITY ASSESSMENT
Each candidate maximal flow velocity waveform, obtained
using a specific grayscale threshold γ ∈ 0, is evaluated on a
beat-by-beat basis via a signal quality assessment algorithm.
The signal quality assessment algorithm is based on thresholding amplitude and duration features as well as template
matching to assess similarity between candidate beats and a
template obtained from the data. This approach is similar to
the method we first presented in [27] and to the recent signal
quality paper by the Neural Analytics Inc (Los Angeles, CA,
USA) group [28].
The first step of our signal quality assessment algorithm
is to detect beat-onsets using the algorithm from [29]. This
algorithm was originally developed for arterial blood pressure
waveforms, but can nonetheless be used to detect beat onsets
in flow velocity waveforms by scaling the input waveform
such that its range matches the range of values expected
for arterial blood pressure (e.g., average peak value around
140 mmHg). Each beat is then subjected to the following
physiological sanity checks
1) Systolic maximum > 30 cm/s.
2) Difference between maximum and minimum flow
velocity in each beat > 20 cm/s.
1) Beat duration > 0.25 s and < 2 s.
2) 0.5 (median beat duration) < beat duration < 2 (median
beat duration).
The rationale behind conditions 1 and 2 is that for the cerebral vessels considered in this work the presumed insonation
angle is < 20◦
. In this case, if we exclude large vessel occlusions, the amplitude and pulsatility of the flow velocity needs
to be sufficiently large to be valid. Very low amplitudes and
lack of pulsatility, generally indicate signal loss (see Fig. 6).
Condition 3 is based on the assumption that the minimal heart
rate is above 30 and below 240 beats per minute, a condition
that is not overly restrictive. Condition 4 excludes excessively
FIGURE 6. Beat-by-beat signal quality assessment of the maximal blood
flow velocity waveform by template matching. The MSE is computed on
the first 75% of the template’s length (zero to black dashed bar). The
candidate beat (shown in green) has a high quality as it matches well to
the template.
short or long beats, that may indicate the presence of false
positives (multiple onsets detected in the same beat) or false
negatives (missed beats). A similar duration-based outlier
detector was used in [30] for improving their cerebral blood
flow velocity pulse-onset detector.
In the next step we compute for each candidate maximal
flow velocity waveform a beat template by taking the median
(with zero-padding of short beats) of all beats of the 1 min
segment that have not been rejected by the physiological
sanity checks described above. Then, each beat that has not
been flagged by the physiological sanity checks is assessed in
terms of the mean squared error (MSE) between the template
(computed using the candidate waveform under examination)
and the candidate beat. To account for heart rate variability,
which mainly affects the diastolic phase of a flow velocity
wavelet, the mean squared error (MSE) is only computed
on the first 75% of the template beat duration (Fig. 6). If
the normalized MSE is above 30%, the beat is flagged as
an artifact. Since in our recording setup we did not expect
large heart rate variability on 1 min segments, we omitted
any linear or non-linear (such as dynamic time warping) beat
alignment.
We define the artifact-index as the percentage of beats that
were labeled as artifacts. For each 1 min segment we then
choose the candidate flow velocity waveform with the smallest artifact-index as our final estimate. Note that, in addition
to allowing us to select a flow velocity waveform (corresponding to a specific grayscale threshold), the signal quality
assessment algorithm provides us with a measure of confidence, a signal quality index (0 % - 100%) for each beat
(Fig. 5 bottom panel). The signal quality index is computed
as one minus the normalized relative MSE between each
beat and the template. Finally, the remaining step to obtain
high-quality flow velocity estimates is post-processing.
