"""Registry of supported experiment readers, kept free of heavy instrument imports."""

EXPERIMENT_READER_MODULES = {
    "akari": "commander4.file_io.experiments.akari",
    "litebird_sim": "commander4.file_io.experiments.litebird_sim",
    "litebird_sim_spawndetectors": (
        "commander4.file_io.experiments.litebird_sim_spawndetectors"
    ),
    "planck_lfi": "commander4.file_io.experiments.planck_lfi",
    "general": "commander4.file_io.experiments.general",
    "SO_LAT": "commander4.file_io.experiments.SO_LAT",
    "SO_SAT": "commander4.file_io.experiments.SO_SAT",
}
