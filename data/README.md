# Data

Nothing in this folder is versioned except this file. The code finds the data through `twin/paths.py`: this folder
by default, or the folder in the environment variable `TWIN_DATA`.

```
data/
  raw/                                                    the rig's files
    PositionTxt_allROI2025-02-20T15_37_59.csv             fish tracking, every frame
    2025.02.20-0957_trajectory_GsizeMotionSingle_180min.csv   the stimulus programme (also in assets/, gzipped)
    ROIdef2025-02-20T12_09_16.csv                         dish geometry on the projector (also in assets/)
    ROIdef.csv                                            the same dishes in camera pixels
  frames/                                                 one table per fish at 30 fps
    2025.02.20_GsizeMotionSingle_f<N>_transformed_abs.csv
  bouts_GsizeMotionSingle.csv                             the segmenter's bout table      (notebook 01)
  bouts_clean.csv                                         the training table              (notebook 01)
  repaired_frames.csv                                     the frames the cleaning repaired (notebook 01)
  schizscreen_twin_v1/                                    the Larsch lab screen (second data set)
  schizscreen_twin_v1.zip
```

`notebooks/01_bout_discretisation.ipynb` rebuilds everything below `raw/`: the 30 fps tables from the three rig files
(`twin/frames.py`; written to `frames/` if that folder is empty, else compared with what is there), then the two bout
tables and the repaired-frame list from `frames/`.

## The experiment

Fifteen larval zebrafish, each alone in a Petri dish 10 cm across, recorded together at 30 frames per second for
174 minutes (313 500 frames). After a pause, the programme alternates 50 s without stimulus and 60 s of stimulus,
95 times: a group of 1, 2, 4, 6 or 8 dots, moving (3.5 cm/s) or static, appears beside the fish, in a 2.5 x 2.5 cm
area at least 0.5 cm from the fish centre. The paradigm is semi-closed-loop: the dots are placed relative to the
fish's position and heading at the onset of the block and then follow the programme.

## The rig's files (`raw/`)

**PositionTxt**: one row per frame, space-separated, no header. The first 45 columns are the fish (x, y and
orientation of each of the 15 fish: absolute camera pixels, origin at the top left, orientation in radians
clockwise); the next 24 the stimulus as displayed (x, y and name of each of 8 dots).

**Trajectory file**: the programme the rig plays (Bonsai), one row per frame, comma-separated, no header, 8 dot slots
x 6 fields (`name, x, y, size, color, background`). `x, y` are in millimetres relative to the fish at block onset,
left = positive x. `name` is the block label, e.g. `CLsemi1800_g0-2_s3.5-3.5` = semi-closed loop, 0 dots on the left
and 2 on the right, speed 3.5 cm/s on each side; `none` between blocks.

**ROIdef**: one row per dish: screen offset (`xoff, yoff`), diameter, centre (`x, y`) and radius, no header.

## The 30 fps tables (`frames/`)

Per fish, one row per frame: `frame`, `stimulus` (the block label, `inter_stim_pause` between blocks), the fish's
`f<N>_x, f<N>_y, f<N>_ori`, and every dot's absolute position and label `d<k>_x, d<k>_y, d<k>_stim` (k = 0-7),
i.e. the tracking and the programme combined in the same camera coordinates. A dot's offset is placed in the dish from
the fish's pose at the first frame of its block; during a pause the dots hold the positions of the block that follows;
y is flipped and angles are counter-clockwise. The tables were first made by the project's earlier frame-level
pipeline (Nayan Adani, 2025); `twin/frames.py` reproduces it exactly, and the stored tables to the last bit of floating
point (positions within 1e-13 px).

## The training table (`bouts_clean.csv`)

One row per bout onset, 71 937 bouts of 15 fish, 69 572 of them active (the bouts every model is fitted on).
`dtheta_deg`, `distance_px` and `dt_s` describe the step from this onset to the next. Positions are camera pixels;
the physical scale is 3.60 px/mm (the profile code's nominal 3.67 px/mm is converted in the figures and tables).

## The second data set (`schizscreen_twin_v1/`)

Unpublished data of the Larsch lab, shared for this collaboration. Its own `DATA_USE.md` sets the terms: keep the
data and anything derived from it out of git and out of any public place. One fish per 70 mm dish, a single dot on
screen, alternating bout-like and continuous episodes; `notebooks/08_second_data_set.ipynb` lists the commands that
process it.
