# Installation

## Requirements

```
cv2==4.12.0
pint-pulsar==1.1.4
```

## Data

Make a symlink called `data` to the directory containing the lightspeed raw data

# Execution

Run the phasing gui with

`python3 main.py OBSERVATION_NAME [args]`

`OBSERVATION_NAME` should be the name you typed into the control gui. Use args to specify which saved data should be used (run `python3 main.py -h` for further information).