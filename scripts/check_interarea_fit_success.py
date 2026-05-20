import os
from os import path

REMOVE_SUCCESS = False


def check_run_integrity(
    rundir, verbose=False, njoblib=40, remove_success_on_incomplete=False
):
    if verbose:
        print(rundir)
    # get chunks
    chunks = [f for f in os.listdir(rundir) if f.startswith("chunk")]
    nchunks = len(chunks)
    if verbose:
        print(f"found {nchunks} chunks.")
    good_chunks = 0
    complete_chunks = 0
    bads = []
    incompletes = []
    for chunk in chunks:
        chunkdir = path.join(rundir, chunk)
        files = os.listdir(path.join(chunkdir))
        if "success" in files:
            good_chunks += 1
        else:
            bads.append(chunk)
        jobfiles = [f for f in files if f.endswith(".joblib")]
        if len(jobfiles) == njoblib:
            complete_chunks += 1
        else:
            incompletes.append(chunk)

        if remove_success_on_incomplete:
            if ("success" in files) and not len(jobfiles) == njoblib:
                successpath = path.join(rundir, chunk, "success")
                os.remove(successpath)

    if verbose:
        print(f"{good_chunks}/{nchunks} are complete.")
    return (good_chunks == nchunks), (complete_chunks == nchunks), bads, incompletes


# list all runs
results_dir = path.join("results", "inter_area")
runs = [
    f for f in os.listdir(results_dir) if not f.startswith(".")
]  # ignore hidden files that may be present

RED = "\033[91m"
STANDARD = "\033[0m"

for r in runs:
    isgood, iscomplete, bads, incompletes = check_run_integrity(
        path.join(results_dir, r), remove_success_on_incomplete=REMOVE_SUCCESS
    )
    if isgood and iscomplete:
        print(r)
    else:
        print(f"{RED}{r} ({len(bads)} bad, {len(incompletes)} incomplete){STANDARD}")
