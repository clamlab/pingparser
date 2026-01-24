# pingparser
python functions for parsing Bonsai raw data outputs (csvs)

the data wrangling is unique to my raw Bonsai output structure:
* (possibly misguided)
* I basically use what I call "event pings"
* while the experiment is trial-based, I don't save experimental params neatly in one
single set per trial
* instead, experimental params arrive row by row in the csv as they are recorded, and can be
at different times within the trial
* hence, the bulk of the wrangling is to put it into a trial by trial format
* in contrast, most other people do this wrangling within Bonsai itself to give a clean trial by trial output
  * e.g. a single Zip containing 20+ parameters
