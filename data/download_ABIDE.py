from nilearn import datasets

datasets.fetch_abide_pcp(
    derivatives='rois_cc200',
    pipeline='cpac',
    band_pass_filtering=True,
    global_signal_regression=True,
    quality_checked=True
)